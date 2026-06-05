#!/usr/bin/env python3
"""
命令行工具
Command Line Interface (CLI)

支持快速评估单张或批量图像
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# 导入核心模块
from quality_index_model import NRIQAModel
from utils import (
    ResultsManager, ConfigManager, BatchProcessor, Logger, FileHandler
)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='NR-IQA: 无参考图像质量评价系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估单张图像
  python cli.py -i image.jpg -p "A beautiful sunset"
  
  # 批量评估
  python cli.py -d ./images -p prompts.txt --output ./results
  
  # 使用自定义配置
  python cli.py -i image.jpg -c config.json
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--image', type=str,
                            help='单张图像路径')
    input_group.add_argument('-d', '--directory', type=str,
                            help='图像目录路径（用于批量处理）')
    
    # 文本提示词
    parser.add_argument('-p', '--prompt', type=str,
                       help='文本提示词或提示词文件路径')
    
    # 配置选项
    parser.add_argument('-c', '--config', type=str,
                       help='配置文件路径（JSON格式）')
    parser.add_argument('-w', '--weights', type=float, nargs=3,
                       metavar=('SEMANTIC', 'TECHNICAL', 'STRUCTURAL'),
                       help='权重配置（需要3个值，和为1）')
    
    # 输出选项
    parser.add_argument('-o', '--output', type=str, default='./results',
                       help='结果输出目录 (default: ./results)')
    parser.add_argument('--format', choices=['json', 'txt', 'both'], 
                       default='json',
                       help='结果输出格式 (default: json)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果')
    
    # 模型选项
    parser.add_argument('--use-clip', action='store_true',
                       help='使用 CLIP 模型进行语义分析')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='计算设备 (default: cpu)')
    
    # 其他选项
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='详细输出模式')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    
    return parser


def load_prompts(prompt_input: Optional[str]) -> Optional[str | List[str]]:
    """
    加载提示词
    
    Args:
        prompt_input: 提示词或文件路径
        
    Returns:
        提示词或提示词列表
    """
    if not prompt_input:
        return None
    
    prompt_path = Path(prompt_input)
    if prompt_path.exists() and prompt_path.is_file():
        # 从文件加载
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    else:
        # 直接返回提示词
        return prompt_input


def process_single_image(image_path: str, prompt: Optional[str], 
                        model: NRIQAModel, logger: Logger) -> dict:
    """
    处理单张图像
    
    Args:
        image_path: 图像路径
        prompt: 文本提示词
        model: NR-IQA 模型
        logger: 日志记录器
        
    Returns:
        评估结果
    """
    from PIL import Image
    import numpy as np
    
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # 评估
        result = model.evaluate(image_array, prompt)
        result['image_path'] = image_path
        
        logger.info(f"✓ 成功评估: {image_path}")
        return result
        
    except Exception as e:
        logger.error(f"✗ 评估失败: {image_path} - {str(e)}")
        return None


def format_result(result: dict, format_type: str = 'txt') -> str:
    """
    格式化评估结果
    
    Args:
        result: 评估结果字典
        format_type: 格式类型 ('txt' 或 'json')
        
    Returns:
        格式化的结果字符串
    """
    if format_type == 'json':
        import json
        from utils import ResultsManager
        json_result = ResultsManager._convert_to_json_compatible(result)
        return json.dumps(json_result, indent=2, ensure_ascii=False)
    
    else:  # txt format
        lines = [
            "=" * 70,
            f"图像: {result.get('image_path', 'Unknown')}",
            "=" * 70,
            "",
            "【综合质量指数】",
            f"  总体分数: {result['overall_quality']:.4f}",
            f"  等级: {result['quality_interpretation']['level']}",
            f"  建议: {result['quality_interpretation']['recommendation']}",
            "",
            "【语义保真度】",
            f"  CLIP 相似度: {result['semantic_fidelity']['clip_similarity']:.4f}",
            f"  关键词匹配率: {result['semantic_fidelity']['keyword_matching_rate']:.4f}",
            f"  保真度指数: {result['semantic_fidelity']['semantic_fidelity']:.4f}",
            "",
            "【技术质量】",
            f"  清晰度: {result['technical_quality']['sharpness']:.4f}",
            f"  噪声: {result['technical_quality']['noise']:.4f}",
            f"  伪影: {result['technical_quality']['artifacts']:.4f}",
            f"  技术质量: {result['technical_quality']['technical_quality']:.4f}",
            "",
            "【结构完整性】",
            f"  边缘连续性: {result['structural_integrity']['edge_continuity']:.4f}",
            f"  形状规则性: {result['structural_integrity']['shape_regularity']:.4f}",
            f"  对象完整性: {result['structural_integrity']['object_completeness']:.4f}",
            f"  边界平滑度: {result['structural_integrity']['boundary_smoothness']:.4f}",
            f"  结构完整性: {result['structural_integrity']['structural_integrity']:.4f}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 初始化日志
    logger = Logger(args.log_file)
    logger.info("NR-IQA 评估系统启动")
    
    try:
        # 加载配置
        config = ConfigManager(args.config)
        
        # 更新权重
        if args.weights:
            if abs(sum(args.weights) - 1.0) > 1e-6:
                logger.error("权重和必须为 1")
                sys.exit(1)
            config.set('semantic_weight', args.weights[0])
            config.set('technical_weight', args.weights[1])
            config.set('structural_weight', args.weights[2])
        
        # 更新其他配置
        if args.use_clip:
            config.set('use_clip', True)
        if args.device:
            config.set('device', args.device)
        
        # 创建模型
        model_config = config.get_model_config()
        model = NRIQAModel(**model_config)
        logger.info("模型初始化完成")
        
        # 创建输出目录
        output_dir = FileHandler.ensure_dir(args.output)
        
        # 加载提示词
        prompts = load_prompts(args.prompt)
        
        # 处理单张图像
        if args.image:
            result = process_single_image(args.image, prompts, model, logger)
            
            if result:
                # 输出结果
                formatted = format_result(result, args.format)
                print(formatted)
                
                # 保存结果
                if not args.no_save:
                    output_file = output_dir / f"{Path(args.image).stem}_result.json"
                    FileHandler.save_json(result, str(output_file))
                    logger.info(f"结果已保存: {output_file}")
        
        # 批量处理
        elif args.directory:
            from glob import glob
            
            # 查找图像文件
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob(str(Path(args.directory) / '**' / ext), 
                                       recursive=True))
            
            if not image_paths:
                logger.warning(f"在 {args.directory} 中未找到图像文件")
                sys.exit(1)
            
            logger.info(f"找到 {len(image_paths)} 张图像")
            
            # 创建批处理器
            processor = BatchProcessor(model, config)
            
            # 处理提示词列表
            prompt_list = None
            if isinstance(prompts, list):
                prompt_list = prompts
            elif prompts:
                prompt_list = [prompts] * len(image_paths)
            
            # 批量处理
            results = processor.process_images(image_paths, prompt_list, 
                                              save_results=not args.no_save)
            
            # 输出统计信息
            logger.info("处理完成")
            stats = processor.get_statistics()
            
            print("\n" + "=" * 70)
            print("批处理统计信息")
            print("=" * 70)
            print(f"  处理图像数: {stats['total_images']}")
            print(f"  综合质量平均值: {stats['overall_mean']:.4f}")
            print(f"  综合质量标准差: {stats['overall_std']:.4f}")
            print(f"  综合质量范围: [{stats['overall_min']:.4f}, {stats['overall_max']:.4f}]")
            print(f"  语义保真度平均值: {stats['semantic_mean']:.4f}")
            print(f"  技术质量平均值: {stats['technical_mean']:.4f}")
            print(f"  结构完整性平均值: {stats['structural_mean']:.4f}")
            print("=" * 70 + "\n")
        
        logger.info("评估完成")
        
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
