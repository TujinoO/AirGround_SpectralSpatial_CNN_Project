"""
Physical-guided MNF module with AMCS component selection.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from hyperspectral_pseudo_label_generator.config import ProcessingConfig


logger = logging.getLogger(__name__)


class PhysicalGuidedPCA:
    def __init__(self, config: ProcessingConfig, wavelengths: np.ndarray):
        self.config = config
        self.wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
        self.pca: Optional[PCA] = None
        self.selected_components: Optional[np.ndarray] = None
        self.scaler_params: Optional[list[Tuple[float, float]]] = None
        self.mean_: Optional[np.ndarray] = None
        self.noise_whitening_: Optional[np.ndarray] = None
        self.noise_dewhitening_: Optional[np.ndarray] = None
        self.mnf_eigenvalues_: Optional[np.ndarray] = None
        self.reconstructed_image_: Optional[np.ndarray] = None
        self.component_corr_: Optional[np.ndarray] = None
        self.component_moran_: Optional[np.ndarray] = None

        if self.wavelengths.shape[0] != 297:
            raise ValueError(f"Expected 297 wavelengths, got {self.wavelengths.shape[0]}")
        if np.any(np.isnan(self.wavelengths)) or np.any(np.isinf(self.wavelengths)):
            raise ValueError("Wavelengths contains NaN or Inf values")

        logger.info(f"Initialized PhysicalGuidedPCA with {config.n_components} components")

    def fit_transform(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D, got {image.ndim}D")

        h, w, bands = image.shape
        if bands != 297:
            raise ValueError(f"Expected 297 bands, got {bands}")
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: ({h}, {w})")
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            raise ValueError("Image contains NaN or Inf values")

        logger.info(f"Fitting MNF on image with shape {image.shape}")

        pixels = image.reshape(-1, bands).astype(np.float64, copy=False)
        self.mean_ = pixels.mean(axis=0)
        centered = pixels - self.mean_

        whitening, dewhitening = self._build_noise_whitening(image)
        self.noise_whitening_ = whitening
        self.noise_dewhitening_ = dewhitening

        whitened_pixels = centered @ whitening

        n_components_full = min(bands, h * w)
        self.pca = PCA(n_components=n_components_full, svd_solver="full")
        self.pca.fit(whitened_pixels)
        scores = self.pca.transform(whitened_pixels)
        self.mnf_eigenvalues_ = np.clip(self.pca.explained_variance_, self.config.epsilon, None)

        self.selected_components = self._select_amcs_components(scores, h=h, w=w, image=image)
        logger.info(f"Selected {len(self.selected_components)} AMCS components")
        logger.info(f"Selected component indices: {self.selected_components}")

        selected_scores = scores[:, self.selected_components]
        transformed = selected_scores.reshape(h, w, -1)
        transformed_scaled = self._scale_components(transformed)

        self.reconstructed_image_ = self._reconstruct_image(scores, h=h, w=w, bands=bands)

        explained_var = self.get_explained_variance()
        total_var = np.sum(explained_var) * 100
        logger.info(f"Selected MNF variance ratio sum: {total_var:.2f}%")

        return transformed_scaled

    def transform_reference(self, spectra: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        missing_state = any(
            (
                self.pca is None,
                self.selected_components is None,
                self.mean_ is None,
                self.noise_whitening_ is None,
            )
        )
        if missing_state:
            raise RuntimeError("Must call fit_transform on image first")
        pca_model = self.pca
        noise_whitening = self.noise_whitening_
        mean = self.mean_
        selected_components = self.selected_components
        assert pca_model is not None
        assert noise_whitening is not None
        assert mean is not None
        assert selected_components is not None

        transformed_spectra: Dict[int, np.ndarray] = {}
        for class_id, spectrum in spectra.items():
            spectrum_arr = np.asarray(spectrum, dtype=np.float64).reshape(-1)
            if spectrum_arr.shape[0] != 297:
                raise ValueError(f"Class {class_id}: expected 297 bands, got {spectrum_arr.shape[0]}")
            if np.any(np.isnan(spectrum_arr)) or np.any(np.isinf(spectrum_arr)):
                raise ValueError(f"Class {class_id}: contains NaN or Inf values")

            centered = spectrum_arr - mean
            whitened = centered @ noise_whitening
            score_full = pca_model.transform(whitened.reshape(1, -1))[0]
            selected = score_full[selected_components]
            transformed_spectra[class_id] = self._scale_single_spectrum(selected)

        return transformed_spectra

    def transform_reference_reconstructed(self, spectra: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        missing_state = any(
            (
                self.pca is None,
                self.selected_components is None,
                self.mean_ is None,
                self.noise_whitening_ is None,
                self.noise_dewhitening_ is None,
            )
        )
        if missing_state:
            raise RuntimeError("Must call fit_transform on image first")
        pca_model = self.pca
        noise_whitening = self.noise_whitening_
        noise_dewhitening = self.noise_dewhitening_
        mean = self.mean_
        selected_components = self.selected_components
        assert pca_model is not None
        assert noise_whitening is not None
        assert noise_dewhitening is not None
        assert mean is not None
        assert selected_components is not None

        reconstructed: Dict[int, np.ndarray] = {}
        for class_id, spectrum in spectra.items():
            spectrum_arr = np.asarray(spectrum, dtype=np.float64).reshape(-1)
            if spectrum_arr.shape[0] != 297:
                raise ValueError(f"Class {class_id}: expected 297 bands, got {spectrum_arr.shape[0]}")
            if np.any(np.isnan(spectrum_arr)) or np.any(np.isinf(spectrum_arr)):
                raise ValueError(f"Class {class_id}: contains NaN or Inf values")

            centered = spectrum_arr - mean
            whitened = centered @ noise_whitening
            score_full = pca_model.transform(whitened.reshape(1, -1))[0]
            filtered = np.zeros_like(score_full)
            filtered[selected_components] = score_full[selected_components]
            whitened_recon = pca_model.inverse_transform(filtered.reshape(1, -1))[0]
            reconstructed_spectrum = whitened_recon @ noise_dewhitening + mean
            reconstructed[class_id] = np.clip(
                reconstructed_spectrum, self.config.epsilon, None
            ).astype(np.float32, copy=False)

        return reconstructed

    def get_reconstructed_image(self) -> np.ndarray:
        if self.reconstructed_image_ is None:
            raise RuntimeError("MNF not fitted yet")
        return self.reconstructed_image_

    def _build_noise_whitening(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diff_h = image[1:, :, :] - image[:-1, :, :]
        diff_w = image[:, 1:, :] - image[:, :-1, :]
        noise_samples = np.concatenate(
            [
                diff_h.reshape(-1, image.shape[2]),
                diff_w.reshape(-1, image.shape[2]),
            ],
            axis=0,
        ).astype(np.float64, copy=False)
        noise_cov = np.cov(noise_samples, rowvar=False)
        noise_cov = noise_cov + self.config.epsilon * np.eye(noise_cov.shape[0], dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(noise_cov)
        eigvals = np.clip(eigvals, self.config.epsilon, None)
        inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        sqrt = np.diag(np.sqrt(eigvals))
        whitening = eigvecs @ inv_sqrt @ eigvecs.T
        dewhitening = eigvecs @ sqrt @ eigvecs.T
        return whitening, dewhitening

    def _select_amcs_components(self, scores: np.ndarray, h: int, w: int, image: np.ndarray) -> np.ndarray:
        if self.mnf_eigenvalues_ is None:
            raise RuntimeError("MNF eigenvalues unavailable")

        pan = image.mean(axis=2).reshape(-1).astype(np.float64, copy=False)
        n_comp = scores.shape[1]
        corr = np.zeros(n_comp, dtype=np.float64)
        moran = np.zeros(n_comp, dtype=np.float64)

        pan_std = float(np.std(pan))
        for i in range(n_comp):
            comp = scores[:, i]
            comp_std = float(np.std(comp))
            if pan_std < self.config.epsilon or comp_std < self.config.epsilon:
                corr[i] = 0.0
            else:
                corr[i] = float(np.corrcoef(comp, pan)[0, 1])
            moran[i] = self._moran_index(comp.reshape(h, w))

        self.component_corr_ = corr
        self.component_moran_ = moran

        shadow_mask = np.abs(corr) >= self.config.amcs_shadow_corr_threshold
        noise_mask = np.logical_and(
            self.mnf_eigenvalues_ <= self.config.amcs_snr_threshold,
            moran <= self.config.amcs_moran_threshold,
        )
        keep_mask = ~(shadow_mask | noise_mask)

        selected = np.where(keep_mask)[0]
        if selected.size < self.config.n_components:
            priority = np.argsort(self.mnf_eigenvalues_)[::-1]
            collected = list(selected.tolist())
            for idx in priority.tolist():
                if idx not in collected:
                    collected.append(idx)
                if len(collected) >= self.config.n_components:
                    break
            selected = np.array(collected, dtype=np.intp)
        elif selected.size > self.config.n_components:
            order = np.argsort(self.mnf_eigenvalues_[selected])[::-1][: self.config.n_components]
            selected = selected[order]

        selected = np.sort(selected.astype(np.intp))
        return selected

    def _reconstruct_image(self, scores: np.ndarray, h: int, w: int, bands: int) -> np.ndarray:
        missing_state = any(
            (
                self.pca is None,
                self.selected_components is None,
                self.mean_ is None,
                self.noise_dewhitening_ is None,
            )
        )
        if missing_state:
            raise RuntimeError("MNF not fitted yet")
        pca_model = self.pca
        noise_dewhitening = self.noise_dewhitening_
        mean = self.mean_
        selected_components = self.selected_components
        assert pca_model is not None
        assert noise_dewhitening is not None
        assert mean is not None
        assert selected_components is not None

        filtered = np.zeros_like(scores)
        filtered[:, selected_components] = scores[:, selected_components]
        whitened_recon = pca_model.inverse_transform(filtered)
        pixels_recon = whitened_recon @ noise_dewhitening + mean
        pixels_recon = np.clip(pixels_recon, self.config.epsilon, None)
        return pixels_recon.reshape(h, w, bands).astype(np.float32, copy=False)

    def _scale_components(self, data: np.ndarray) -> np.ndarray:
        _, _, k = data.shape
        scaled = np.zeros_like(data, dtype=np.float32)
        self.scaler_params = []

        for idx in range(k):
            component = data[:, :, idx].astype(np.float64, copy=False)
            min_val = float(component.min())
            max_val = float(component.max())
            if max_val - min_val < self.config.epsilon:
                scaled[:, :, idx] = 0.0
                self.scaler_params.append((0.0, 1.0))
            else:
                scaled[:, :, idx] = ((component - min_val) / (max_val - min_val)).astype(np.float32, copy=False)
                self.scaler_params.append((min_val, max_val))

        return scaled

    def _scale_single_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        if self.scaler_params is None:
            raise RuntimeError("Scaler parameters not initialized")

        scaled = np.zeros_like(spectrum, dtype=np.float32)
        for idx, (min_val, max_val) in enumerate(self.scaler_params):
            if max_val - min_val < self.config.epsilon:
                scaled[idx] = 0.0
            else:
                value = (spectrum[idx] - min_val) / (max_val - min_val)
                scaled[idx] = float(np.clip(value, 0.0, 1.0))
        return scaled

    def _moran_index(self, component_map: np.ndarray) -> float:
        values = component_map.astype(np.float64, copy=False)
        mean_val = float(values.mean())
        dev = values - mean_val
        denom = float(np.sum(dev * dev))
        if denom < self.config.epsilon:
            return 0.0

        horizontal = float(np.sum(dev[:, 1:] * dev[:, :-1]))
        vertical = float(np.sum(dev[1:, :] * dev[:-1, :]))
        numerator = horizontal + vertical

        n = values.size
        w_sum = (values.shape[0] * (values.shape[1] - 1)) + ((values.shape[0] - 1) * values.shape[1])
        if w_sum <= 0:
            return 0.0

        return (n / w_sum) * (numerator / denom)

    def get_explained_variance(self) -> np.ndarray:
        if self.mnf_eigenvalues_ is None or self.selected_components is None:
            raise RuntimeError("PCA not fitted yet")
        total = float(np.sum(self.mnf_eigenvalues_))
        if total <= self.config.epsilon:
            return np.zeros(len(self.selected_components), dtype=np.float32)
        return (self.mnf_eigenvalues_[self.selected_components] / total).astype(np.float32, copy=False)
