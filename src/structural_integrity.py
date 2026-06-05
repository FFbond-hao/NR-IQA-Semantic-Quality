"""
结构完整性指标计算模块
Structural Integrity Analysis Module

基于边缘连续性和形状规则性的结构完整性评估
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage
from scipy.ndimage import label, find_objects
import warnings


class StructuralIntegrityAnalyzer:
    """
    结构完整性分析器
    
    评估图像的结构质量：
    1. 边缘连续性 (Edge Continuity)
    2. 形状规则性 (Shape Regularity)
    3. 对象完整性 (Object Completeness)
    """
    
    def __init__(self, edge_threshold: float = 0.1):
        """
        初始化结构完整性分析器
        
        Args:
            edge_threshold: 边缘检测阈值
        """
        self.edge_threshold = edge_threshold
        self.sobel_x = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]], dtype=np.float32)
    
    @staticmethod
    def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
        """RGB 转灰度"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            return gray
        return image
    
    def compute_edge_continuity(self, image: np.ndarray) -> float:
        """
        计算边缘连续性指标
        
        数学定义:
        EdgeContinuity = (连续边缘长度) / (总边缘长度)
        
        连续性度量:
        - 完全连续的边缘：值接近 1
        - 断裂的边缘：值低于 0.5
        - 无边缘：值为 0
        
        物理意义：
        - 高值表示图像中的对象边界清晰连贯
        - 低值表示边缘存在明显断裂，可能是伪影或低质量
        
        Args:
            image: 输入图像
            
        Returns:
            边缘连续性 ∈ [0, 1]
        """
        gray = self._rgb_to_gray(image)
        
        # Sobel 边缘检测
        gx = ndimage.convolve(gray.astype(np.float32), self.sobel_x)
        gy = ndimage.convolve(gray.astype(np.float32), self.sobel_y)
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        # 边缘二值化
        threshold = np.mean(gradient_magnitude) + np.std(gradient_magnitude)
        edges = (gradient_magnitude > threshold).astype(np.uint8)
        
        if np.sum(edges) < 1:
            return 0.0
        
        # 连续性分析：检查相邻像素的连通性
        # 使用形态学操作进行连接
        from scipy.ndimage import binary_closing, binary_dilation
        
        edges_closed = binary_closing(edges, iterations=1)
        edges_dilated = binary_dilation(edges_closed, iterations=1)
        
        # 标记连通分量
        labeled, num_components = label(edges_dilated)
        
        if num_components == 0:
            return 0.0
        
        # 计算最大连通分量的长度
        component_sizes = np.bincount(labeled.flatten())
        max_component_size = np.max(component_sizes[1:]) if len(component_sizes) > 1 else 0
        
        # 计算连续性比例
        continuity = max_component_size / np.sum(edges)
        
        return float(np.clip(continuity, 0.0, 1.0))
    
    def compute_shape_regularity(self, image: np.ndarray) -> float:
        """
        计算形状规则性指标
        
        数学定义:
        Regularity = (对象周长²) / (4π * 面积)
        这是圆形度指标，也称为 Polsby-Popper 指数
        
        理论范围:
        - 完全圆形：值 = 1
        - 正方形：值 ≈ 0.9
        - 不规则形状：值 < 0.5
        
        物理意义：
        - 高值表示图像中的形状规则、边界平滑
        - 低值表示形状不规则、边界粗糙（可能是伪影）
        
        Args:
            image: 输入图像
            
        Returns:
            形状规则性 ∈ [0, 1]
        """
        gray = self._rgb_to_gray(image)
        
        # Otsu 阈值分割
        threshold = np.mean(gray) + np.std(gray)
        binary = (gray > threshold).astype(np.uint8)
        
        # 获取轮廓
        from scipy import ndimage as ndi
        labeled, num_objects = label(binary)
        
        if num_objects == 0:
            return 0.5  # 无法计算时返回中等值
        
        regularity_scores = []
        
        # 对每个对象计算规则性
        for obj_id in range(1, num_objects + 1):
            obj_mask = (labeled == obj_id).astype(np.uint8)
            
            # 计算面积和周长
            area = np.sum(obj_mask)
            
            # 周长通过边界像素数估计
            edges = ndi.sobel(obj_mask.astype(np.float32))
            perimeter = np.sum(edges > 0)
            
            if area < 1 or perimeter < 1:
                continue
            
            # Polsby-Popper 圆形度指标
            # 完美圆形的值为 1，不规则形状的值较小
            regularity = perimeter ** 2 / (4 * np.pi * area) if area > 0 else 0
            regularity = 1.0 / (1.0 + regularity)  # 反转使得圆形为高值
            
            regularity_scores.append(regularity)
        
        if not regularity_scores:
            return 0.5
        
        # 返回平均规则性
        mean_regularity = np.mean(regularity_scores)
        return float(np.clip(mean_regularity, 0.0, 1.0))
    
    def compute_object_completeness(self, image: np.ndarray) -> float:
        """
        计算对象完整性指标
        
        数学定义:
        Completeness = (不存在孔洞的对象比例)
        
        评估方法：
        - 检测对象内部是否存在孔洞
        - 孔洞表示对象的缺失或损坏
        
        物理意义：
        - 高值表示对象形状完整、无损坏
        - 低值表示对象存在明显孔洞或破损
        
        Args:
            image: 输入图像
            
        Returns:
            对象完整性 ∈ [0, 1]
        """
        gray = self._rgb_to_gray(image)
        
        # 二值化
        threshold = np.mean(gray) + np.std(gray)
        binary = (gray > threshold).astype(np.uint8)
        
        # 标记对象
        from scipy.ndimage import label, binary_fill_holes
        labeled, num_objects = label(binary)
        
        if num_objects == 0:
            return 0.5
        
        # 检测孔洞
        complete_count = 0
        
        for obj_id in range(1, num_objects + 1):
            obj_mask = (labeled == obj_id).astype(np.uint8)
            
            # 填充孔洞
            filled = binary_fill_holes(obj_mask).astype(np.uint8)
            
            # 计算孔洞面积
            hole_area = np.sum(filled) - np.sum(obj_mask)
            obj_area = np.sum(obj_mask)
            
            if obj_area < 1:
                continue
            
            # 如果孔洞面积小于对象面积的 10%，认为对象完整
            if hole_area / obj_area < 0.1:
                complete_count += 1
        
        completeness = complete_count / num_objects if num_objects > 0 else 0.5
        
        return float(np.clip(completeness, 0.0, 1.0))
    
    def compute_boundary_smoothness(self, image: np.ndarray) -> float:
        """
        计算边界平滑度指标
        
        数学定义:
        Smoothness = 1 - mean(|边界曲率|)
        
        曲率衡量：
        - 平滑曲线：低曲率
        - 锯齿状边界：高曲率
        
        Args:
            image: 输入图像
            
        Returns:
            边界平滑度 ∈ [0, 1]
        """
        gray = self._rgb_to_gray(image)
        
        # 边缘检测
        edges = ndimage.sobel(gray.astype(np.float32))
        edge_binary = (edges > np.mean(edges)).astype(np.uint8)
        
        # 计算边界的曲率变化
        # 使用拉普拉斯算子检测曲率
        laplacian = ndimage.laplace(gray.astype(np.float32))
        
        # 高曲率区域
        high_curvature_region = np.abs(laplacian) > np.percentile(np.abs(laplacian), 75)
        
        if np.sum(edge_binary) < 1:
            return 0.5
        
        # 边界上的高曲率比例
        curvature_ratio = np.sum(high_curvature_region & (edge_binary > 0)) / np.sum(edge_binary)
        
        smoothness = 1.0 - curvature_ratio
        
        return float(np.clip(smoothness, 0.0, 1.0))
    
    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """
        完整的结构完整性分析流程
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, 3)
            
        Returns:
            包含各项指标的字典
        """
        results = {}
        
        # 计算单个指标
        results['edge_continuity'] = self.compute_edge_continuity(image)
        results['shape_regularity'] = self.compute_shape_regularity(image)
        results['object_completeness'] = self.compute_object_completeness(image)
        results['boundary_smoothness'] = self.compute_boundary_smoothness(image)
        
        # 计算综合结构完整性指标
        # Integrity = α*EdgeContinuity + β*ShapeRegularity + γ*Completeness + δ*Smoothness
        alpha, beta, gamma, delta = 0.3, 0.25, 0.25, 0.2
        
        structural_integrity = (
            alpha * results['edge_continuity'] +
            beta * results['shape_regularity'] +
            gamma * results['object_completeness'] +
            delta * results['boundary_smoothness']
        )
        results['structural_integrity'] = structural_integrity
        
        # 添加权重信息
        results['weights'] = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
        
        return results
    
    def interpret_integrity_score(self, score: float) -> str:
        """
        解释结构完整性分数
        
        Args:
            score: 完整性分数 ∈ [0, 1]
            
        Returns:
            解释文本
        """
        if score >= 0.8:
            return "优秀 - 边缘清晰、形状规则、对象完整"
        elif score >= 0.6:
            return "良好 - 结构较完整，边界连贯"
        elif score >= 0.4:
            return "中等 - 结构完整性一般，存在轻微缺陷"
        elif score >= 0.2:
            return "较差 - 边界不连贯，形状不规则"
        else:
            return "很差 - 严重的结构问题或损坏"


# 示例使用
if __name__ == "__main__":
    analyzer = StructuralIntegrityAnalyzer()
    
    # 创建示例图像
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    results = analyzer.analyze(image)
    
    print("结构完整性分析结果:")
    for key, value in results.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value:.4f}")
    
    print(f"\n解释: {analyzer.interpret_integrity_score(results['structural_integrity'])}")
