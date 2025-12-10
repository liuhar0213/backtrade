#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器

统一加载和验证所有配置文件
"""
import json
from pathlib import Path
from typing import Dict, Any
import sys
import io

# Avoid reassigning stdout at import time (breaks pytest terminal I/O).
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ConfigLoader:
    """配置加载器"""

    def __init__(self, root_dir: str = None):
        """初始化"""
        if root_dir is None:
            root_dir = Path(__file__).parent.parent
        self.root_dir = Path(root_dir)

        self.configs = {}

    def load_all(self) -> Dict[str, Any]:
        """加载所有配置"""
        print("=" * 80)
        print("配置加载器 - ABCDE系统")
        print("=" * 80)

        # 加载paths.json（优先）
        paths = self.load_json("config/paths.json")
        self.configs['paths'] = paths

        # 加载其他配置
        config_files = {
            'manifest': 'config/manifest.json',
            'system_state': 'config/system_state.json',
            'feature_catalog': 'A_knowledge/feature_catalog.json',
            'feature_config': 'B_features/feature_config.json',
            'fusion_config': 'C_fusion/fusion_config.json',
            'execution_config': 'E_execution/execution_config.json',
            'supervisor_config': 'D_supervisor/supervisor_config.json'
        }

        print(f"\n根目录: {self.root_dir}")
        print(f"\n加载配置文件:")

        for name, path in config_files.items():
            try:
                config = self.load_json(path)
                self.configs[name] = config
                print(f"  ✓ {name:20s} -> {path}")
            except Exception as e:
                print(f"  ✗ {name:20s} -> {path} (错误: {e})")

        return self.configs

    def load_json(self, rel_path: str) -> Dict[str, Any]:
        """加载JSON文件"""
        full_path = self.root_dir / rel_path

        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def validate(self) -> bool:
        """验证配置完整性"""
        print("\n" + "=" * 80)
        print("配置验证")
        print("=" * 80)

        required_configs = [
            'manifest', 'paths', 'system_state',
            'feature_catalog', 'feature_config',
            'fusion_config', 'execution_config',
            'supervisor_config'
        ]

        all_valid = True

        for name in required_configs:
            if name in self.configs:
                print(f"  ✓ {name} 已加载")
            else:
                print(f"  ✗ {name} 缺失")
                all_valid = False

        # 验证关键字段
        print("\n关键字段验证:")

        # 验证manifest
        if 'manifest' in self.configs:
            manifest = self.configs['manifest']
            required_fields = ['version', 'description', 'data_period']
            for field in required_fields:
                if field in manifest:
                    print(f"  ✓ manifest.{field}: {manifest[field] if not isinstance(manifest[field], dict) else '...'}")
                else:
                    print(f"  ✗ manifest.{field} 缺失")
                    all_valid = False

        # 验证execution_config
        if 'execution_config' in self.configs:
            exec_cfg = self.configs['execution_config']
            if 'strategy' in exec_cfg:
                strategy = exec_cfg['strategy']
                print(f"  ✓ 黄金配置: bias_threshold={strategy.get('bias_threshold')}, "
                      f"atr_multiplier={strategy.get('atr_multiplier')}")

        # 验证paths
        if 'paths' in self.configs:
            paths = self.configs['paths']
            if 'data' in paths:
                data = paths['data']
                print(f"  ✓ 数据源: {data.get('symbol')} @ {data.get('timeframe')}")

        print("\n" + "=" * 80)
        if all_valid:
            print("✓ 所有配置验证通过！")
        else:
            print("✗ 配置验证失败，请检查缺失项")
        print("=" * 80)

        return all_valid

    def get(self, key: str, default=None):
        """获取配置"""
        return self.configs.get(key, default)

    def summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 80)
        print("配置摘要")
        print("=" * 80)

        if 'manifest' in self.configs:
            manifest = self.configs['manifest']
            print(f"\n版本: {manifest.get('version')}")
            print(f"描述: {manifest.get('description')}")

            if 'performance_baseline' in manifest:
                perf = manifest['performance_baseline']
                print(f"\n黄金配置基线性能:")
                print(f"  总收益: {perf.get('total_return')}%")
                print(f"  Sharpe: {perf.get('sharpe_ratio')}")
                print(f"  胜率: {perf.get('win_rate')}%")
                print(f"  最大回撤: {perf.get('max_drawdown')}%")
                print(f"  交易数: {perf.get('total_trades')}笔")

        if 'feature_catalog' in self.configs:
            catalog = self.configs['feature_catalog']
            if 'domains' in catalog:
                print(f"\n特征域: {list(catalog['domains'].keys())}")
                total_features = sum(len(domain['features']) for domain in catalog['domains'].values())
                print(f"特征总数: {total_features}")

        if 'fusion_config' in self.configs:
            fusion = self.configs['fusion_config']
            if 'buckets' in fusion:
                buckets = fusion['buckets']
                print(f"\n融合桶: {list(buckets.keys())}")
                for bucket_name, bucket_cfg in buckets.items():
                    print(f"  {bucket_name}: weight={bucket_cfg.get('bucket_weight')}, "
                          f"features={len(bucket_cfg.get('features', {}))}")

        if 'supervisor_config' in self.configs:
            supervisor = self.configs['supervisor_config']
            enabled_modules = []
            for module in ['drift_monitor', 'coherence_monitor', 'f_infinity', 'param_advisor']:
                if supervisor.get(module, {}).get('enabled'):
                    enabled_modules.append(module)
            print(f"\n监督模块: {enabled_modules}")

        print("\n" + "=" * 80)


def main():
    """主函数"""
    loader = ConfigLoader()

    # 加载所有配置
    configs = loader.load_all()

    # 验证
    valid = loader.validate()

    # 摘要
    if valid:
        loader.summary()

    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
