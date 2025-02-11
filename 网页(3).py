import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from xgboost import XGBClassifier
import xgboost as xgb

# 添加复杂的 CSS 样式，修复背景颜色问题
st.markdown("""
    <style>
  .main {
        background-color: #1F75FE;
        background-image: url('https://www.transparenttextures.com/patterns/bedge-grunge.png');
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
  .title {
        font-size: 48px;
        color: #144999;
        text-align: center;
        margin-bottom: 30px;
    }
  .subheader {
        font-size: 28px;
        color: #FFD700;
        margin-bottom: 25px;
        text-align: center;
        border-bottom: 2px solid #FFA500;
        padding-bottom: 10px;
        margin-top: 20px;
    }
  .input-label {
        font-size: 18px;
        font-weight: bold;
        color: #FFA500;
        margin-bottom: 10px;
    }
  .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 16px;
        color: #87CEEB;
        background-color: #144999;
        padding: 20px;
        border-top: 1px solid #87CEFA;
    }
  .button {
        background-color: #FFA500;
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 20px auto;
        cursor: pointer;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
        transition: background-color 0.3s, box-shadow 0.3s;
    }
  .button:hover {
        background-color: #FF8C00;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.7);
    }
  .stSelectbox,.stNumberInput,.stSlider {
        margin-bottom: 20px;
    }
  .stSlider > div {
        padding: 10px;
        background: #D0E0E3;
        border-radius: 10px;
    }
  .prediction-result {
        font-size: 24px;
        color: #ffffff;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background: #87CEFA;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    }
  .advice-text {
        font-size: 20px;
        line-height: 1.6;
        color: #ffffff;
        background: #FFA500;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# 页面标题
st.markdown('<div class="title">职业紧张预测 SHAP 分析</div>', unsafe_allow_html=True)

# 加载 XGBoost 模型
try:
    model = joblib.load('best_model_1.pkl')
except Exception as e:
    st.write(f"<div style='color: red;'>Error loading model: {e}</div>", unsafe_allow_html=True)
    model = None

# 获取模型输入特征数量及顺序
model_input_features = ['A3', 'A5', 'B2', 'B3', 'B4', 'B5', 'exerciseG1', '工龄', '生活满意度', '抑郁症状级别', '睡眠状况']
expected_feature_count = len(model_input_features)

# 定义新的特征选项及名称
category_mapping = {0: '无职业紧张症状', 1: '轻度职业紧张症状', 2: '中度职业紧张症状', 3: '重度职业紧张症状'}

# Streamlit 界面设置
st.markdown('<div class="subheader">请填写以下信息以进行职业紧张预测：</div>', unsafe_allow_html=True)

# A3（学历）选择
A3_options = {1: '初中及以下', 2: '高中或中专', 3: '大专或高职', 4: '大学本科', 5: '研究生及以上'}
A3 = st.selectbox("您的最高学历：", options=list(A3_options.keys()), format_func=lambda x: A3_options[x])

# A5（月收入）选择
A5_options = {1: '少于 3000 元', 2: '3000 - 4999 元', 3: '5000 - 6999 元', 4: '7000 - 8999 元', 5: '9000 - 10999 元', 6: '11000 元及以上'}
A5 = st.selectbox("您的实际平均月收入：", options=list(A5_options.keys()), format_func=lambda x: A5_options[x])

# 工龄输入
service_years = st.number_input("您的工龄为：", min_value=1, max_value=120, value=10)

# B2（近一个月以来平均每周工作天数）输入
work_days_per_week = st.number_input("近一个月以来，您平均每周工作天数约为：", min_value=1, max_value=7, value=5)

# B3（近一个月以来平均每天加班时间）输入
overtime_hours = st.number_input("近一个月以来，您平均每天加班时间约为：（请填写）______ 小时", min_value=0, max_value=24, value=0)

# B4（是否轮班）选择
B4_options = {1: '否', 2: '是'}
B4 = st.selectbox("您的工作是否为轮班工作制？", options=list(B4_options.keys()), format_func=lambda x: B4_options[x])

# B5（是否需要上夜班）选择
B5_options = {1: '否', 2: '是'}
B5 = st.selectbox("您的工作是否需要上夜班？", options=list(B5_options.keys()), format_func=lambda x: B5_options[x])

# G1（外出步行或骑自行车情况）选择
g1_options = {1: '无', 2: '偶尔，1-3 次/月', 3: '有，1~3 次/周', 4: '经常，4~6 次/周', 5: '每天'}
g1 = st.selectbox("您在外出时，是否有步行或骑自行车持续至少 30 分钟的情况？", options=list(g1_options.keys()), format_func=lambda x: g1_options[x])

# 生活满意度滑块
life_satisfaction = st.slider("您的生活满意度评分为（1 - 5）：", min_value=1, max_value=5, value=3)

# 睡眠状况滑块
sleep_status = st.slider("您的睡眠状况评分为（1 - 5）：", min_value=1, max_value=5, value=3)

# 抑郁症状级别滑块
depression_level = st.slider("您的抑郁症状评分为（1 - 5）：", min_value=1, max_value=5, value=3)

def predict():
    try:
        # 检查模型是否加载成功
        if model is None:
            st.write("<div style='color: red;'>模型加载失败，无法进行预测。</div>", unsafe_allow_html=True)
            return

        # 获取用户输入并构建特征数组
        user_inputs = {
            'A3': int(A3),
            'A5': int(A5),
            'B2': int(work_days_per_week),
            'B3': int(overtime_hours),
            'B4': int(B4),
            'B5': int(B5),
            'exerciseG1': int(g1),
            '工龄': int(service_years),
            '生活满意度': int(life_satisfaction),
            '抑郁症状级别': int(depression_level),
            '睡眠状况': int(sleep_status),
        }

        feature_values = [user_inputs[feature] for feature in model_input_features]
        features_array = np.array([feature_values])

        # 使用 XGBoost 模型进行预测
        predicted_class = model.predict(features_array)[0]
        predicted_proba = model.predict_proba(features_array)[0]

        # 显示预测结果
        st.markdown(f"<div class='prediction-result'>预测类别：{category_mapping[predicted_class]}</div>", unsafe_allow_html=True)

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        advice = {
            '无职业紧张症状': f"根据我们的模型，该员工无职业紧张症状。模型预测该员工无职业紧张症状的概率为 {probability:.2f}%。请继续保持良好的工作和生活状态。",
            '轻度职业紧张症状': f"根据我们的模型，该员工有轻度职业紧张症状。模型预测该员工职业紧张程度为轻度的概率为 {probability:.2f}%。建议您适当调整工作节奏，关注自身身心健康。",
            '中度职业紧张症状': f"根据我们的模型，该员工有中度职业紧张症状。模型预测该员工职业紧张程度为中度的概率为 {probability:.2f}%。建议您寻求专业帮助，如心理咨询或与上级沟通调整工作。",
            '重度职业紧张症状': f"根据我们的模型，该员工有重度职业紧张症状。模型预测该员工职业紧张程度为重度的概率为 {probability:.2f}%。强烈建议您立即采取行动，如休假、寻求医疗支持或与管理层协商改善工作环境。",
        }[category_mapping[predicted_class]]
        st.markdown(f"<div class='advice-text'>{advice}</div>", unsafe_allow_html=True)

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_array)

        # 计算每个类别的特征贡献度
        importance_df = pd.DataFrame()
        for i in range(shap_values.shape[2]):  # 对每个类别进行计算
            importance = np.abs(shap_values[:, :, i]).mean(axis=0)
            importance_df[f'Class_{i}'] = importance

        importance_df.index = model_input_features

        # 类别映射
        type_mapping = {
            0: '无职业紧张症状',
            1: '轻度职业紧张症状',
            2: '中度职业紧张症状',
            3: '重度职业紧张症状',
        }
        importance_df.columns = [type_mapping[i] for i in range(importance_df.shape[1])]

        # 获取指定类别的 SHAP 值贡献度
        predicted_class_name = category_mapping[predicted_class]  # 根据预测类别获取类别名称
        importances = importance_df[predicted_class_name]  # 提取 importance_df 中对应的类别列

        # 准备绘制瀑布图的数据
        feature_name_mapping = {
            'A3': '婚姻状况',
            'A5': '月均收入',
            'B2': '近一个月每周工作天数',
            'B3': '近一个月每日加班时间',
            'B4': '是否轮班',
            'B5': '是否夜班',
            'exerciseG1': '锻炼情况',
            '工龄': '工龄',
            '生活满意度': '生活满意度',
            '抑郁症状级别': '抑郁症状级别',
            '睡眠状况': '睡眠状况'
        }
        features = [feature_name_mapping[f] for f in importances.index.tolist()]  # 获取特征名称
        contributions = importances.values  # 获取特征贡献度

        # 确保瀑布图的数据是按贡献度绝对值降序排列的
        sorted_indices = np.argsort(np.abs(contributions))[::-1]
        features_sorted = [features[i] for i in sorted_indices]
        contributions_sorted = contributions[sorted_indices]

        # 初始化绘图
        fig, ax = plt.subplots(figsize=(14, 8))

        # 初始化累积值
        start = 0
        prev_contributions = [start]  # 起始值为0

        # 计算每一步的累积值
        for i in range(1, len(contributions_sorted)):
            prev_contributions.append(prev_contributions[-1] + contributions_sorted[i - 1])

        font_prop = FontProperties(family='SimHei')  # 可以根据需要调整字体和大小

        # 绘制瀑布图
        for i in range(len(contributions_sorted)):
            color = '#ff5050' if contributions_sorted[i] < 0 else '#66b3ff'  # 负贡献使用红色，正贡献使用蓝色
            if i == len(contributions_sorted) - 1:
                # 最后一个条形带箭头效果，表示最终累积值
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5, hatch='/')
            else:
                ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5)

            # 在每个条形上显示数值
            plt.text(prev_contributions[i] + contributions_sorted[i] / 2, i, f"{contributions_sorted[i]:.2f}", 
                    ha='center', va='center', fontsize=10, fontproperties=font_prop, color='black')

        # 设置图表属性
        plt.title(f'{predicted_class_name} 的特征贡献度瀑布图', fontsize=18, fontproperties=font_prop)
        plt.xlabel('贡献度 (SHAP 值)', fontsize=14, fontproperties=font_prop)
        plt.ylabel('特征', fontsize=14, fontproperties=font_prop)
        plt.yticks(fontsize=12, fontproperties=font_prop)
        plt.xticks(fontsize=12, fontproperties=font_prop)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # 增加边距避免裁剪
        plt.xlim(left=0, right=max(prev_contributions) + max(contributions_sorted) * 1.0)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

        plt.tight_layout()

        # 保存并在 Streamlit 中展示
        plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_waterfall_plot.png")

    except Exception as e:
        st.write(f"<div style='color: red;'>Error in prediction: {e}</div>", unsafe_allow_html=True)

if st.button("预测", key="predict_button"):
    predict()

# 页脚
st.markdown('<div class="footer">© 2024 All rights reserved.</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
