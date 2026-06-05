"""
示例使用和最佳实践
Example Usage and Best Practices
"""

import numpy as np
from pathlib import Path
import sys

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quality_index_model import NRIQAModel
from utils import ConfigManager, ResultsManager, BatchProcessor, Logger
from PIL import Image


def example_1_basic_usage():
    """示例 1: 基本使用 - 评估单张图像"""
    print("\n" + "=" * 70)
    print("示例 1: 基本使用 - 评估单张图像")
    print("=" * 70 + "\n")
    
    # 初始化模型（使用默认配置）
    model = NRIQAModel(
        weight_semantic=0.35,
        weight_technical=0.40,
        weight_structural=0.25
    )
    
    # 创建示例图像
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    test_prompt = "A beautiful sunset over the ocean with golden light"
    
    # 评估图像
    result = model.evaluate(test_image, test_prompt)
    
    # 显示结果
    print("评估结果:")
    print(f"  综合质量指数: {result['overall_quality']:.4f}")
    print(f"  等级: {result['quality_interpretation']['level']}")
    print(f"  语义保真度: {result['semantic_fidelity']['semantic_fidelity']:.4f}")
    print(f"  技术质量: {result['technical_quality']['technical_quality']:.4f}")
    print(f"  结构完整性: {result['structural_integrity']['structural_integrity']:.4f}")
    
    # 生成报告
    report = model.generate_report(result)
    print("\n详细报告:")
    print(report)


def example_2_custom_weights():
    """示例 2: 自定义权重配置"""
    print("\n" + "=" * 70)
    print("示例 2: 自定义权重配置")
    print("=" * 70 + "\n")
    
    # 不同应用场景的权重配置
    configs = {
        '文本驱动生成': (0.5, 0.3, 0.2),
        '通用评估': (0.35, 0.40, 0.25),
        '物体检测': (0.2, 0.3, 0.5),
        '艺术创意': (0.3, 0.35, 0.35)
    }
    
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_prompt = "A cat on a table"
    
    print("不同配置的评分对比:\n")
    print(f"{'配置名称':<15} {'语义':>8} {'技术':>8} {'结构':>8} {'综合':>8}")
    print("-" * 50)
    
    for config_name, (w1, w2, w3) in configs.items():
        model = NRIQAModel(weight_semantic=w1, weight_technical=w2, 
                          weight_structural=w3)
        result = model.evaluate(test_image, test_prompt)
        
        print(f"{config_name:<15} {w1:>8.2f} {w2:>8.2f} {w3:>8.2f} " +
              f"{result['overall_quality']:>8.4f}")


def example_3_batch_processing():
    """示例 3: 批量处理"""
    print("\n" + "=" * 70)
    print("示例 3: 批量处理")
    print("=" * 70 + "\n")
    
    # 创建模型
    model = NRIQAModel()
    
    # 创建配置管理器
    config = ConfigManager()
    
    # 创建结果管理器
    results_manager = ResultsManager('./results')
    
    # 创建多张测试图像
    test_images = [
        np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    test_prompts = [
        "A beautiful sunset",
        "A cozy room",
        "Mountain landscape"
    ]
    
    print("处理 3 张图像...\n")
    
    for idx, (image, prompt) in enumerate(zip(test_images, test_prompts), 1):
        result = model.evaluate(image, prompt)
        results_manager.add_result(result, f'image_{idx}.jpg')
        print(f"✓ 图像 {idx}: 质量指数 = {result['overall_quality']:.4f}")
    
    # 保存结果
    results_manager.save_all()
    
    # 显示统计信息
    stats = results_manager.get_summary_stats()
    print("\n统计信息:")
    print(f"  处理图像数: {stats['total_images']}")
    print(f"  质量平均值: {stats['overall_mean']:.4f}")
    print(f"  质量标准差: {stats['overall_std']:.4f}")
    print(f"  质量范围: [{stats['overall_min']:.4f}, {stats['overall_max']:.4f}]")


def example_4_configuration_management():
    """示例 4: 配置管理"""
    print("\n" + "=" * 70)
    print("示例 4: 配置管理")
    print("=" * 70 + "\n")
    
    # 创建配置管理器
    config = ConfigManager()
    
    print("默认配置:")
    print(f"  语义权重: {config.get('semantic_weight')}")
    print(f"  技术权重: {config.get('technical_weight')}")
    print(f"  结构权重: {config.get('structural_weight')}")
    print(f"  使用 CLIP: {config.get('use_clip')}")
    print(f"  设备: {config.get('device')}")
    print(f"  批大小: {config.get('batch_size')}")
    
    # 修改配置
    config.set('semantic_weight', 0.5)
    config.set('technical_weight', 0.3)
    config.set('structural_weight', 0.2)
    config.set('use_clip', True)
    
    print("\n修改后的配置:")
    print(f"  语义权重: {config.get('semantic_weight')}")
    print(f"  技术权重: {config.get('technical_weight')}")
    print(f"  结构权重: {config.get('structural_weight')}")
    print(f"  使用 CLIP: {config.get('use_clip')}")
    
    # 保存配置
    config.save('./config.json')
    print("\n配置已保存到: ./config.json")
    
    # 获取模型配置
    model_config = config.get_model_config()
    print(f"\n模型配置: {model_config}")


def example_5_logging():
    """示例 5: 日志记录"""
    print("\n" + "=" * 70)
    print("示例 5: 日志记录")
    print("=" * 70 + "\n")
    
    # 创建日志记录器
    logger = Logger('./logs/example.log')
    
    logger.info("开始处理图像")
    logger.info("加载模型...")
    
    # 模拟处理
    try:
        logger.info("处理图像 1...")
        logger.info("✓ 图像 1 处理完成")
        
        logger.info("处理图像 2...")
        logger.warning("图像 2 质量较低")
        
        logger.info("处理完成")
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
    
    print("✓ 日志已保存到: ./logs/example.log")


def example_6_error_handling():
    """示例 6: 错误处理"""
    print("\n" + "=" * 70)
    print("示例 6: 错误处理")
    print("=" * 70 + "\n")
    
    logger = Logger()
    model = NRIQAModel()
    
    # 测试 1: 权重验证
    print("测试 1: 验证权重和为 1")
    try:
        invalid_model = NRIQAModel(weight_semantic=0.5, weight_technical=0.3, 
                                  weight_structural=0.3)
        logger.error("应该抛出异常!")
    except AssertionError as e:
        logger.info(f"✓ 正确捕获异常: {str(e)}")
    
    # 测试 2: 无效的图像类型
    print("\n测试 2: 无效的图像类型")
    try:
        invalid_image = "not an image"
        result = model.evaluate(invalid_image, "test")
        logger.error("应该抛出异常!")
    except (AttributeError, TypeError) as e:
        logger.info("✓ 正确捕获异常")
    
    print("\n✓ 所有错误处理测试通过")


def example_7_best_practices():
    """示例 7: 最佳实践"""
    print("\n" + "=" * 70)
    print("示例 7: 最佳实践")
    print("=" * 70 + "\n")
    
    print("""
【NR-IQA 系统使用最佳实践】

1. 模型初始化
   ✓ 根据应用场景选择合适的权重配置
   ✓ 如需使用 CLIP，确保已安装 torch 和 clip
   ✓ 根据硬件情况选择计算设备 (cpu/cuda/mps)

2. 图像处理
   ✓ 确保输入图像格式正确 (RGB/灰度, dtype=uint8)
   ✓ 图像尺寸可变，建议 256x256 以上
   ✓ 避免过度压缩导致伪影过多

3. 文本提示词
   ✓ 提示词应具体明确，避免歧义
   ✓ 包含主体、属性、场景、风格等信息
   ✓ 过长的提示词可能影响关键词匹配

4. 结果解释
   ✓ 综合质量指数 > 0.8: 优秀，可直接使用
   ✓ 综合质量指数 0.6-0.8: 良好，基本满足要求
   ✓ 综合质量指数 < 0.6: 需要优化改进

5. 批量处理
   ✓ 使用 BatchProcessor 进行大规模评估
   ✓ 使用 ConfigManager 管理统一的配置
   ✓ 使用 Logger 记录处理过程

6. 性能优化
   ✓ 对大量图像时启用结果缓存
   ✓ 考虑使用 GPU 加速（如果可用）
   ✓ 合理设置批处理大小

7. 扩展和定制
   ✓ 可继承分析器类进行自定义
   ✓ 可调整权重系数适应特定任务
   ✓ 可添加新的质量指标维度

8. 故障排查
   ✓ 检查输入图像格式和大小
   ✓ 验证权重配置（和为1）
   ✓ 查看日志获取详细错误信息
    """)


def main():
    """运行所有示例"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  NR-IQA 系统使用示例和最佳实践".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    examples = [
        ("基本使用", example_1_basic_usage),
        ("自定义权重", example_2_custom_weights),
        ("批量处理", example_3_batch_processing),
        ("配置管理", example_4_configuration_management),
        ("日志记录", example_5_logging),
        ("错误处理", example_6_error_handling),
        ("最佳实践", example_7_best_practices),
    ]
    
    print("可用示例:")
    for idx, (name, _) in enumerate(examples, 1):
        print(f"  {idx}. {name}")
    
    print("\n运行所有示例...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ 示例执行错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "█" * 70)
    print("█" + "所有示例执行完成".center(70) + "█")
    print("█" * 70 + "\n")


if __name__ == '__main__':
    main()
