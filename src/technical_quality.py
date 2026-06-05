"""
技术质量指标计算模块
Technical Quality Analysis Module

基于自然场景统计(NSS)特征和频域分析的技术质量评估
包括: 清晰度、噪声、伪影等指标
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage
from scipy.fft import fft2, fftshift
import warnings


class TechnicalQualityAnalyzer:
    """
    技术质量分析器
    
    评估图像的客观质量指标：
    1. 清晰度 (Sharpness)
    2. 噪声水平 (Noise)
    3. 伪影 (Artifacts)
    """
    
    def __init__(self):
        """初始化技术质量分析器"""
        self.laplacian_kernel = np.array([[0, -1, 0],
                                          [-1, 4, -1],
                                          [0, -1, 0]], dtype=np.float32)
    
    @staticmethod
    def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
        """
        RGB 转灰度
        
        Args:
            image: RGB 图像 shape (H, W, 3) 或灰度 (H, W)
            
        Returns:
            灰度图像 shape (H, W)
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 标准灰度转换公式
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            return gray
        return image
    
    def compute_sharpness(self, image: np.ndarray) -> float:
        """
        计算图像清晰度指标
        
        数学定义:
        Sharpness = mean(|∇²I|²)
        其中 ∇² 是拉普拉斯算子
        
        统计意义：
        - 高清晰度：梯度变化大，熵高
        - 低清晰度：梯度变化小，熵低
        - 范围: [0, +∞)，归一化后 [0, 1]
        
        Args:
            image: 输入图像
            
        Returns:
            清晰度分数 ∈ [0, 1]
        """
        gray = self._rgb_to_gray(image)
        
        # 应用拉普拉斯算子
        laplacian = ndimage.convolve(gray.astype(np.float32), self.laplacian_kernel)
        
        # 计算 Laplacian 的方差作为清晰度指标
        sharpness_score = np.var(laplacian)
        
        # 归一化到 [0, 1]
        # 使用 sigmoid 函数进行软归一化
        normalized_sharpness = 1.0 / (1.0 + np.exp(-sharpness_score / 1000.0))
        
        return float(np.clip(normalized_sharpness, 0.0, 1.0))
    
    def compute_noise(self, image: np.ndarray) -> float:
        """
        计算图像噪声水平
        
        数学定义:
        基于 Gabor 滤波的高频能量分析
        Noise = E_high_freq / E_total
        
        其中:
        - E_high_freq: 高频分量能量
        - E_total: 总能量
        
        统计意义：
        - 高噪声：高频分量多，比值高
        - 低噪声：高频分量少，比值低
        - 范围: [0, 1]
        
        Args:
            image: 输入图像
            
        Returns:
            噪声水平 ∈ [0, 1] (低值表示噪声少)
        """
        gray = self._rgb_to_gray(image)
        
        # FFT 变换
        fft_result = fft2(gray)
        fft_shift = fftshift(fft_result)
        
        # 幅度谱
        magnitude = np.abs(fft_shift)
        
        # 计算总能量
        total_energy = np.sum(magnitude ** 2)
        
        if total_energy < 1e-10:
            return 0.5
        
        # 高频分量：使用中心距离作为频率估计
        h, w = magnitude.shape
        y = np.arange(h) - h // 2
        x = np.arange(w) - w // 2
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        # 距离中心大于总距离30%的区域认为是高频
        distance = np.sqrt(xx ** 2 + yy ** 2)
        max_distance = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
        high_freq_mask = distance > 0.3 * max_distance
        
        high_freq_energy = np.sum((magnitude * high_freq_mask) ** 2)
        
        # 噪声指标：高频能量占比
        noise_ratio = high_freq_energy / total_energy
        
        return float(np.clip(noise_ratio, 0.0, 1.0))
    
    def compute_artifacts(self, image: np.ndarray) -> float:
        """
        计算图像伪影指标
        
        数学定义:
        Artifacts = mean(edge_discontinuity) + mean(frequency_anomaly)
        
        伪影特征:
        - 不自然的边缘断裂
        - 频域中的异常峰值（块状伪影）
        - 不连续的色阶变化
        
        Args:
            image: 输入图像
            
        Returns:
            伪影强度 ∈ [0, 1] (低值表示伪影少)
        """
        gray = self._rgb_to_gray(image)
        
        # 方法1: 边缘连续性分析
        # 使用 Sobel 算子计算梯度
        from scipy.ndimage import sobel
        
        gx = sobel(gray, axis=0)
        gy = sobel(gray, axis=1)
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        # 计算梯度的方差变化（检测不自然的边缘断裂）
        local_var = ndimage.generic_filter(gradient_magnitude, np.var, size=5)
        edge_discontinuity = np.std(local_var)
        
        # 方法2: 频域异常检测（块状伪影）
        fft_result = fft2(gray)
        fft_shift = fftshift(fft_result)
        magnitude = np.abs(fft_shift)
        
        # 检测频域中的峰值（块状伪影特征）
        # 计算频谱的峰值与中间值的比值
        h, w = magnitude.shape
        center = magnitude[h//4:3*h//4, w//4:3*w//4]
        edges = np.concatenate([
            magnitude[0:h//4, :].flatten(),
            magnitude[3*h//4:, :].flatten()
        ])
        
        center_mean = np.mean(center) if center.size > 0 else 1e-10
        edges_mean = np.mean(edges) if edges.size > 0 else 1e-10
        
        frequency_anomaly = edges_mean / (center_mean + 1e-10)
        frequency_anomaly = np.clip(frequency_anomaly, 0, 1)
        
        # 综合伪影指标
        artifacts = (edge_discontinuity + frequency_anomaly) / 2.0
        
        # 归一化
        normalized_artifacts = 1.0 / (1.0 + np.exp(-artifacts))
        
        return float(np.clip(normalized_artifacts, 0.0, 1.0))
    
    def compute_nss_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        计算自然场景统计(NSS)特征
        
        NSS 特征包括:
        1. 对数归一化直方图
        2. 色彩分布
        3. 局部对比度分布
        
        Args:
            image: 输入图像
            
        Returns:
            NSS 特征字典
        """
        gray = self._rgb_to_gray(image)
        
        features = {}
        
        # 特征1: 灰度直方图熵
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        features['entropy'] = entropy / 8.0  # 归一化到 [0, 1]
        
        # 特征2: 对比度
        contrast = np.std(gray)
        features['contrast'] = np.clip(contrast / 128.0, 0, 1)
        
        # 特征3: 局部活动性 (local activity)
        lx = np.convolve(gray.flatten(), [-1, 1], mode='valid')
        ly = np.convolve(gray.flatten(), [-1, 1], mode='valid')
        local_activity = np.mean(np.abs(lx)) + np.mean(np.abs(ly))
        features['local_activity'] = np.clip(local_activity / 256.0, 0, 1)
        
        # 特征4: 亮度水平
        brightness = np.mean(gray) / 256.0
        features['brightness'] = brightness
        
        return features
    
    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """
        完整的技术质量分析流程
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, 3)
            
        Returns:
            包含各项质量指标的字典
        """
        results = {}
        
        # 计算单个指标
        results['sharpness'] = self.compute_sharpness(image)
        results['noise'] = self.compute_noise(image)
        results['artifacts'] = self.compute_artifacts(image)
        
        # 计算 NSS 特征
        nss_features = self.compute_nss_features(image)
        results.update(nss_features)
        
        # 计算综合技术质量指标
        # Quality = α*Sharpness + β*(1-Noise) + γ*(1-Artifacts)
        alpha, beta, gamma = 0.4, 0.3, 0.3
        
        technical_quality = (
            alpha * results['sharpness'] +
            beta * (1.0 - results['noise']) +
            gamma * (1.0 - results['artifacts'])
        )
        results['technical_quality'] = technical_quality
        
        # 添加权重信息
        results['weights'] = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        
        return results
    
    def interpret_quality_score(self, score: float) -> str:
        """
        解释技术质量分数
        
        Args:
            score: 质量分数 ∈ [0, 1]
            
        Returns:
            解释文本
        """
        if score >= 0.8:
            return "优秀 - 高清晰度、低噪声、无伪影"
        elif score >= 0.6:
            return "良好 - 较好的清晰度和噪声控制"
        elif score >= 0.4:
            return "中等 - 清晰度和伪影控制一般"
        elif score >= 0.2:
            return "较差 - 明显的清晰度问题或噪声"
        else:
            return "很差 - 严重的质量问题"


# 示例使用
if __name__ == "__main__":
    analyzer = TechnicalQualityAnalyzer()
    
    # 创建示例图像
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    results = analyzer.analyze(image)
    
    print("技术质量分析结果:")
    for key, value in results.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value:.4f}")
    
    print(f"\n解释: {analyzer.interpret_quality_score(results['technical_quality'])}")
