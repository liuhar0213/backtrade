"""
快速验证F5 SHAP修复

测试SHAP特征选择器是否能正确处理numpy数组输入
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("F5 SHAP API 修复验证")
print("=" * 60)

# 生成测试数据（numpy数组格式）
print("\n[1] 生成测试数据...")
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    random_state=42
)
print(f"  数据类型: {type(X)}")
print(f"  数据形状: {X.shape}")
print(f"  标签分布: {np.bincount(y)}")

# 训练模型
print("\n[2] 训练RandomForest模型...")
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X, y)
print(f"  训练完成，准确率: {model.score(X, y):.4f}")

# 测试SHAP特征选择
print("\n[3] 测试SHAP特征选择...")
try:
    from F_intelligence.shap_feature_selector import SHAPFeatureSelector
    from config_manager import ABCDEFConfigManager

    config = ABCDEFConfigManager()
    selector = SHAPFeatureSelector(
        config_manager=config,
        model=model,
        model_type='tree',
        verbose=1
    )

    # 使用numpy数组（而非DataFrame）
    import pandas as pd
    X_small = X[:200]  # 使用部分数据加速
    y_small = y[:200]

    result = selector.select_features(
        X=pd.DataFrame(X_small),  # 先测试DataFrame
        y=pd.Series(y_small),
        n_features=5
    )

    print(f"\n[DataFrame测试] [SUCCESS]")
    print(f"  选择特征数: {result.n_selected_features}")
    print(f"  前3个特征: {result.selected_features[:3]}")

    # 现在测试numpy数组
    print("\n[4] 测试numpy数组输入...")
    result2 = selector.select_features(
        X=X_small,  # 直接传numpy数组
        y=y_small,
        n_features=5
    )

    print(f"\n[numpy数组测试] [SUCCESS]")
    print(f"  选择特征数: {result2.n_selected_features}")
    print(f"  前3个特征: {result2.selected_features[:3]}")

    print("\n" + "=" * 60)
    print("[PASS] F5 SHAP API 修复验证通过!")
    print("=" * 60)
    print("\n修复内容:")
    print("  1. 支持numpy数组和DataFrame输入")
    print("  2. 正确处理SHAP分数的类型转换（使用.item()）")
    print("  3. 自动生成特征名称（feature_0, feature_1, ...）")
    print("  4. 自动修复维度不匹配（截取正确数量）")

except ImportError as e:
    print(f"\n[WARNING] SHAP库未安装: {e}")
    print("  安装命令: pip install shap")

except Exception as e:
    print(f"\n[FAILED] 测试失败: {e}")
    import traceback
    traceback.print_exc()
