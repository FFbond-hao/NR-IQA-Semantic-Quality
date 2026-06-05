"""
综合质量指数模型
Overall Quality Index Model

整合语义保真度、技术质量、结构完整性三个维度
建立加权综合质量评估体系
"""

import numpy as np
from typing import Dict, Optional, Tuple
import json
from .text_semantic_parser import TextSemanticParser
from .semantic_fidelity import SemanticFidelityAnalyzer
from .technical_quality import TechnicalQualityAnalyzer
from .structural_integrity import StructuralIntegrityAnalyzer


class NRIQAModel:
    """
    无参考图像质量评价模型 (No-Reference Image Quality Assessment)
    
    综合质量指数数学模型:
    
    Q_final = w₁ * S_fidelity + w₂ * Q_technical + w₃ * I_structural
    
    其中:
    - w₁, w₂, w₃: 权重系数 (Σw_i = 1)
    - S_fidelity: 语义保真度 ∈ [0, 1]
    - Q_technical: 技术质量 ∈ [0, 1]
    - I_structural: 结构完整性 ∈ [0, 1]
    
    物理意义:
    - Q_final ∈ [0, 1]: 综合质量指数
    - 高值 (>0.7): 优秀质量
    - 中值 (0.4-0.7): 一般到良好
    - 低值 (<0.4): 质量较差
    """
    
    def __init__(self, 
                 weight_semantic: float = 0.35,
                 weight_technical: float = 0.40,
                 weight_structural: float = 0.25,
                 use_clip: bool = False):
        """
        初始化 NR-IQA 模型
        
        Args:
            weight_semantic: 语义保真度权重
            weight_technical: 技术质量权重
            weight_structural: 结构完整性权重
            use_clip: 是否使用 CLIP 模型（需要额外依赖）
        """
        # 验证权重和为 1
        total_weight = weight_semantic + weight_technical + weight_structural
        assert abs(total_weight - 1.0) < 1e-6, "权重和必须为 1"
        
        self.weight_semantic = weight_semantic
        self.weight_technical = weight_technical
        self.weight_structural = weight_structural
        
        # 初始化子模块
        self.text_parser = TextSemanticParser()
        self.semantic_analyzer = SemanticFidelityAnalyzer()
        self.technical_analyzer = TechnicalQualityAnalyzer()
        self.structural_analyzer = StructuralIntegrityAnalyzer()
        
        if use_clip:
            self.semantic_analyzer.initialize_clip()
    
    def evaluate(self, image: np.ndarray, 
                prompt: Optional[str] = None,
                image_features: Optional[np.ndarray] = None,
                text_features: Optional[np.ndarray] = None,
                image_description: Optional[str] = None,
                detected_objects: Optional[list] = None) -> Dict:
        """
        完整的图像质量评价流程
        
        Args:
            image: 输入图像
            prompt: 文本提示词（用于语义保真度评估）
            image_features: 图像特征向量（CLIP）
            text_features: 文本特征向量（CLIP）
            image_description: 图像描述（用于关键词匹配）
            detected_objects: 检测到的对象列表
            
        Returns:
            包含所有评价指标的字典
        """
        results = {
            'timestamp': np.datetime64('now').astype(str),
            'model_version': '1.0'
        }
        
        # ===== 第一阶段: 文本语义解析 =====
        if prompt:
            semantic_parse = self.text_parser.parse(prompt)
            results['semantic_parse'] = {
                k: {'tokens': v.tokens, 'confidence': v.confidence}
                for k, v in semantic_parse.items()
            }
            
            # 提取关键词用于后续分析
            keywords = {k: v.tokens for k, v in semantic_parse.items()}
        else:
            keywords = None
            results['semantic_parse'] = None
        
        # ===== 第二阶段: 语义保真度分析 =====
        semantic_results = self.semantic_analyzer.analyze(
            image_features=image_features,
            text_features=text_features,
            prompt_keywords=keywords,
            image_description=image_description,
            detected_objects=detected_objects
        )
        results['semantic_fidelity'] = semantic_results
        semantic_fidelity_score = semantic_results['semantic_fidelity']
        
        # ===== 第三阶段: 技术质量分析 =====
        technical_results = self.technical_analyzer.analyze(image)
        results['technical_quality'] = technical_results
        technical_quality_score = technical_results['technical_quality']
        
        # ===== 第四阶段: 结构完整性分析 =====
        structural_results = self.structural_analyzer.analyze(image)
        results['structural_integrity'] = structural_results
        structural_integrity_score = structural_results['structural_integrity']
        
        # ===== 第五阶段: 综合质量指数计算 =====
        overall_quality = self._compute_overall_quality(
            semantic_fidelity_score,
            technical_quality_score,
            structural_integrity_score
        )
        
        results['overall_quality'] = overall_quality
        results['quality_interpretation'] = self._interpret_overall_score(overall_quality)
        
        # ===== 第六阶段: 模型参数和解释 =====
        results['model_parameters'] = {
            'weights': {
                'semantic': self.weight_semantic,
                'technical': self.weight_technical,
                'structural': self.weight_structural
            },
            'formula': 'Q_final = w₁*S_fidelity + w₂*Q_technical + w₃*I_structural'
        }
        
        return results
    
    def _compute_overall_quality(self, semantic: float, 
                                technical: float, 
                                structural: float) -> float:
        """
        计算综合质量指数
        
        数学模型:
        Q_final = w₁ * S_fidelity + w₂ * Q_technical + w₃ * I_structural
        
        其中权重满足: w₁ + w₂ + w₃ = 1
        
        权重的物理意义:
        - w₁ (semantic): 文本-图像一致性的重要性
          * 0.35: 适中，用于一般场景
          * 0.5+: 强调语义一致性（如文本驱动生成）
          * 0.2-: 适用于无特定提示的图像评估
        
        - w₂ (technical): 客观质量指标的重要性
          * 0.40: 标准权重，平衡主观和客观
          * 0.5+: 强调客观清晰度和噪声
          * 0.3-: 当关注内容而非技术细节时
        
        - w₃ (structural): 对象完整性和边界质量的重要性
          * 0.25: 标准权重
          * 0.3+: 强调边界质量（如物体检测任务）
          * 0.15-: 当关注整体印象而非细节时
        
        Args:
            semantic: 语义保真度 ∈ [0, 1]
            technical: 技术质量 ∈ [0, 1]
            structural: 结构完整性 ∈ [0, 1]
            
        Returns:
            综合质量指数 ∈ [0, 1]
        """
        overall = (
            self.weight_semantic * semantic +
            self.weight_technical * technical +
            self.weight_structural * structural
        )
        
        return float(np.clip(overall, 0.0, 1.0))
    
    def _interpret_overall_score(self, score: float) -> Dict[str, str]:
        """
        解释综合质量分数
        
        评分标准：
        - 0.80-1.00: 优秀 - 适合高端应用（广告、出版、专业制作）
        - 0.60-0.80: 良好 - 适合一般应用（网络、社交媒体）
        - 0.40-0.60: 中等 - 需要改进（概念验证、内部使用）
        - 0.20-0.40: 较差 - 不建议发布（测试、原型）
        - 0.00-0.20: 很差 - 需要重新生成
        
        Args:
            score: 综合质量分数
            
        Returns:
            包含等级和建议的字典
        """
        if score >= 0.8:
            level = "优秀 (Excellent)"
            recommendation = "可用于高端应用、商业发布"
            action = "approve"
        elif score >= 0.6:
            level = "良好 (Good)"
            recommendation = "适合一般应用、网络发布"
            action = "approve_with_minor"
        elif score >= 0.4:
            level = "中等 (Fair)"
            recommendation = "建议进行优化或微调"
            action = "review"
        elif score >= 0.2:
            level = "较差 (Poor)"
            recommendation = "不建议直接使用，需要改进"
            action = "reject_with_guidance"
        else:
            level = "很差 (Very Poor)"
            recommendation = "建议重新生成或调整参数"
            action = "reject"
        
        return {
            'level': level,
            'score': score,
            'recommendation': recommendation,
            'action': action
        }
    
    def batch_evaluate(self, images: list, prompts: Optional[list] = None) -> list:
        """
        批量评估多张图像
        
        Args:
            images: 图像列表
            prompts: 对应的文本提示词列表
            
        Returns:
            评估结果列表
        """
        results = []
        
        for idx, image in enumerate(images):
            prompt = prompts[idx] if prompts and idx < len(prompts) else None
            result = self.evaluate(image, prompt)
            results.append(result)
        
        return results
    
    def generate_report(self, eval_result: Dict) -> str:
        """
        生成评估报告
        
        Args:
            eval_result: 单个图像的评估结果
            
        Returns:
            格式化的报告字符串
        """
        report = "=" * 60 + "\n"
        report += "无参考图像质量评价 (NR-IQA) 报告\n"
        report += "=" * 60 + "\n\n"
        
        # 综合质量指数
        report += "【综合质量指数】\n"
        overall = eval_result['overall_quality']
        interpretation = eval_result['quality_interpretation']
        report += f"  总体分数: {overall:.4f}\n"
        report += f"  等级: {interpretation['level']}\n"
        report += f"  建议: {interpretation['recommendation']}\n\n"
        
        # 语义保真度
        report += "【语义保真度】\n"
        sem = eval_result['semantic_fidelity']
        report += f"  CLIP 相似度: {sem['clip_similarity']:.4f}\n"
        report += f"  关键词匹配率: {sem['keyword_matching_rate']:.4f}\n"
        report += f"  保真度指数: {sem['semantic_fidelity']:.4f}\n\n"
        
        # 技术质量
        report += "【技术质量】\n"
        tech = eval_result['technical_quality']
        report += f"  清晰度: {tech['sharpness']:.4f}\n"
        report += f"  噪声: {tech['noise']:.4f}\n"
        report += f"  伪影: {tech['artifacts']:.4f}\n"
        report += f"  技术质量: {tech['technical_quality']:.4f}\n\n"
        
        # 结构完整性
        report += "【结构完整性】\n"
        struct = eval_result['structural_integrity']
        report += f"  边缘连续性: {struct['edge_continuity']:.4f}\n"
        report += f"  形状规则性: {struct['shape_regularity']:.4f}\n"
        report += f"  对象完整性: {struct['object_completeness']:.4f}\n"
        report += f"  边界平滑度: {struct['boundary_smoothness']:.4f}\n"
        report += f"  结构完整性: {struct['structural_integrity']:.4f}\n\n"
        
        report += "=" * 60 + "\n"
        
        return report
    
    def export_results(self, eval_result: Dict, format: str = 'json') -> str:
        """
        导出评估结果
        
        Args:
            eval_result: 评估结果字典
            format: 导出格式 ('json', 'dict')
            
        Returns:
            格式化的结果字符串
        """
        if format == 'json':
            # 转换为 JSON 兼容格式
            json_compatible = self._convert_to_json_compatible(eval_result)
            return json.dumps(json_compatible, indent=2)
        elif format == 'dict':
            return str(eval_result)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _convert_to_json_compatible(obj):
        """转换对象为 JSON 兼容格式"""
        if isinstance(obj, dict):
            return {k: NRIQAModel._convert_to_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [NRIQAModel._convert_to_json_compatible(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return float(obj)
        else:
            return obj


if __name__ == "__main__":
    # 示例使用
    print("NR-IQA 模型演示")
    print("-" * 60)
    
    # 创建示例图像
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    test_prompt = "A beautiful sunset over the ocean"
    
    # 初始化模型
    model = NRIQAModel(
        weight_semantic=0.35,
        weight_technical=0.40,
        weight_structural=0.25
    )
    
    # 评估图像
    result = model.evaluate(test_image, test_prompt)
    
    # 生成报告
    report = model.generate_report(result)
    print(report)
