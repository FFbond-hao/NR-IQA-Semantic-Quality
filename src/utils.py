"""
实用工具函数模块
Utility Functions Module

包含数据处理、文件操作、结果管理等辅助功能
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import pickle


class FileHandler:
    """文件处理工具"""
    
    @staticmethod
    def ensure_dir(directory: str) -> Path:
        """
        确保目录存在
        
        Args:
            directory: 目录路径
            
        Returns:
            Path 对象
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def save_json(data: Dict, filepath: str) -> None:
        """
        保存为 JSON 文件
        
        Args:
            data: 数据字典
            filepath: 保存路径
        """
        FileHandler.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ 已保存: {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """
        从 JSON 文件加载
        
        Args:
            filepath: 文件路径
            
        Returns:
            数据字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, filepath: str) -> None:
        """
        保存为 pickle 文件
        
        Args:
            data: 数据对象
            filepath: 保存路径
        """
        FileHandler.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ 已保存: {filepath}")
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        从 pickle 文件加载
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的数据对象
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ResultsManager:
    """评估结果管理工具"""
    
    def __init__(self, output_dir: str = './results'):
        """
        初始化结果管理器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = FileHandler.ensure_dir(output_dir)
        self.results = []
    
    def add_result(self, result: Dict, image_name: Optional[str] = None) -> None:
        """
        添加评估结果
        
        Args:
            result: 评估结果字典
            image_name: 图像名称
        """
        if image_name:
            result['image_name'] = image_name
        result['timestamp'] = datetime.now().isoformat()
        self.results.append(result)
    
    def save_all(self, filename: str = 'evaluation_results.json') -> Path:
        """
        保存所有结果
        
        Args:
            filename: 文件名
            
        Returns:
            保存路径
        """
        filepath = self.output_dir / filename
        
        # 转换为 JSON 兼容格式
        json_compatible = []
        for result in self.results:
            json_result = self._convert_to_json_compatible(result)
            json_compatible.append(json_result)
        
        FileHandler.save_json({'results': json_compatible}, str(filepath))
        return filepath
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        获取评估结果的统计信息
        
        Returns:
            统计信息字典
        """
        if not self.results:
            return {}
        
        overall_scores = [r.get('overall_quality', 0) for r in self.results]
        semantic_scores = [r.get('semantic_fidelity', {}).get('semantic_fidelity', 0) 
                          for r in self.results]
        technical_scores = [r.get('technical_quality', {}).get('technical_quality', 0) 
                           for r in self.results]
        structural_scores = [r.get('structural_integrity', {}).get('structural_integrity', 0) 
                            for r in self.results]
        
        return {
            'overall_mean': np.mean(overall_scores),
            'overall_std': np.std(overall_scores),
            'overall_min': np.min(overall_scores),
            'overall_max': np.max(overall_scores),
            'semantic_mean': np.mean(semantic_scores),
            'technical_mean': np.mean(technical_scores),
            'structural_mean': np.mean(structural_scores),
            'total_images': len(self.results)
        }
    
    @staticmethod
    def _convert_to_json_compatible(obj: Any) -> Any:
        """转换为 JSON 兼容格式"""
        if isinstance(obj, dict):
            return {k: ResultsManager._convert_to_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ResultsManager._convert_to_json_compatible(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return float(obj)
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)


class ConfigManager:
    """配置管理工具"""
    
    DEFAULT_CONFIG = {
        'semantic_weight': 0.35,
        'technical_weight': 0.40,
        'structural_weight': 0.25,
        'use_clip': False,
        'clip_model': 'ViT-B/32',
        'device': 'cpu',
        'batch_size': 8,
        'quality_thresholds': {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        if config_file and os.path.exists(config_file):
            self.config = FileHandler.load_json(config_file)
        else:
            self.config = self.DEFAULT_CONFIG.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.config[key] = value
    
    def update(self, updates: Dict) -> None:
        """批量更新配置"""
        self.config.update(updates)
    
    def save(self, filepath: str) -> None:
        """保存配置文件"""
        FileHandler.save_json(self.config, filepath)
    
    def get_model_config(self) -> Dict:
        """获取模型配置"""
        return {
            'weight_semantic': self.get('semantic_weight'),
            'weight_technical': self.get('technical_weight'),
            'weight_structural': self.get('structural_weight'),
            'use_clip': self.get('use_clip')
        }


class BatchProcessor:
    """批量处理工具"""
    
    def __init__(self, model, config: Optional[ConfigManager] = None):
        """
        初始化批处理器
        
        Args:
            model: NR-IQA 模型实例
            config: 配置管理器
        """
        self.model = model
        self.config = config or ConfigManager()
        self.results_manager = ResultsManager()
    
    def process_images(self, image_paths: List[str], prompts: Optional[List[str]] = None,
                      save_results: bool = True) -> List[Dict]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像文件路径列表
            prompts: 对应的文本提示词列表
            save_results: 是否保存结果
            
        Returns:
            评估结果列表
        """
        from PIL import Image
        
        results = []
        
        for idx, image_path in enumerate(image_paths):
            try:
                # 加载图像
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
                
                # 获取提示词
                prompt = prompts[idx] if prompts and idx < len(prompts) else None
                
                # 评估
                result = self.model.evaluate(image_array, prompt)
                result['image_path'] = image_path
                
                # 添加到管理器
                self.results_manager.add_result(result, os.path.basename(image_path))
                results.append(result)
                
                print(f"✓ 已处理: {image_path} ({idx+1}/{len(image_paths)})")
                
            except Exception as e:
                print(f"✗ 处理失败: {image_path} - {str(e)}")
        
        # 保存结果
        if save_results:
            self.results_manager.save_all()
        
        return results
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.results_manager.get_summary_stats()


class PerformanceMetrics:
    """性能评估指标"""
    
    @staticmethod
    def compute_correlation(predictions: List[float], 
                           ground_truth: List[float]) -> Tuple[float, float]:
        """
        计算皮尔逊相关系数
        
        Args:
            predictions: 预测值列表
            ground_truth: 真实值列表
            
        Returns:
            相关系数和 p 值
        """
        from scipy.stats import pearsonr
        
        if len(predictions) != len(ground_truth):
            raise ValueError("预测值和真实值长度不匹配")
        
        correlation, p_value = pearsonr(predictions, ground_truth)
        return correlation, p_value
    
    @staticmethod
    def compute_mae(predictions: List[float], 
                   ground_truth: List[float]) -> float:
        """
        计算平均绝对误差
        
        Args:
            predictions: 预测值列表
            ground_truth: 真实值列表
            
        Returns:
            MAE 值
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        return float(np.mean(np.abs(predictions - ground_truth)))
    
    @staticmethod
    def compute_rmse(predictions: List[float], 
                    ground_truth: List[float]) -> float:
        """
        计算均方根误差
        
        Args:
            predictions: 预测值列表
            ground_truth: 真实值列表
            
        Returns:
            RMSE 值
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        return float(np.sqrt(np.mean((predictions - ground_truth) ** 2)))
    
    @staticmethod
    def compute_spearman_rank(predictions: List[float], 
                             ground_truth: List[float]) -> Tuple[float, float]:
        """
        计算斯皮尔曼秩相关系数
        
        Args:
            predictions: 预测值列表
            ground_truth: 真实值列表
            
        Returns:
            斯皮尔曼相关系数和 p 值
        """
        from scipy.stats import spearmanr
        
        spearman_corr, p_value = spearmanr(predictions, ground_truth)
        return spearman_corr, p_value


class Logger:
    """日志记录工具"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        if log_file:
            FileHandler.ensure_dir(os.path.dirname(log_file))
    
    def log(self, level: str, message: str) -> None:
        """
        记录日志
        
        Args:
            level: 日志级别 (INFO, WARNING, ERROR)
            message: 日志消息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
    
    def info(self, message: str) -> None:
        """记录 INFO 级别"""
        self.log('INFO', message)
    
    def warning(self, message: str) -> None:
        """记录 WARNING 级别"""
        self.log('WARNING', message)
    
    def error(self, message: str) -> None:
        """记录 ERROR 级别"""
        self.log('ERROR', message)
