import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('LGB1-7.pkl')

# 定义特征名称
feature_names = [
    "2965", "6567", "11641", "3929", "4767", "13637", "9869"
]

# Streamlit 用户界面
st.title("Salmonella Antimicrobial Resistance Prediction App")

# 用户输入特征数据
input_2965 = st.number_input("2965±1:", min_value=0.000000001, max_value=1.0, value=0.000200916, format="%.9f")
input_6567 = st.number_input("6567±1:", min_value=0.0000000001, max_value=1.0, value=0.0000521, format="%.9f")
input_11641 = st.number_input("11641±1:", min_value=0.000000001, max_value=1.0, value=0.00011864, format="%.9f")
input_3929 = st.number_input("3929±1:", min_value=0.0000000001, max_value=1.0, value=0.000149248, format="%.9f")
input_4767 = st.number_input("4767±1:", min_value=0.0000000001, max_value=1.0, value=0.0000521, format="%.9f")
input_13637 = st.number_input("13637±1:", min_value=0.000000001, max_value=1.0, value=0.00011864, format="%.9f")
input_9869 = st.number_input("9869±1:", min_value=0.0000000001, max_value=1.0, value=0.000149248, format="%.9f")

# 将输入的数据转化为模型的输入格式
feature_values = [
    input_2965, input_6567, input_11641, input_3929, input_4767, input_13637, input_9869
]
features = np.array([feature_values])

# 创建 SHAP 解释器
explainer = shap.TreeExplainer(model)

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (0: Susceptible, 1: Resistant)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, the Salmonella is predicted to be resistant. "
            f"The probability of the Salmonella being predicted as a resistant strain by the model is {probability:.1f}%. "
        )
    else:
        advice = (
            f"According to our model, the Salmonella is predicted to be susceptible. "
            f"The probability of the Salmonella being predicted as a susceptible strain by the model is {probability:.1f}%. "
        )

    st.write(advice)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(features)

    # 显示 SHAP waterfall 图
    st.subheader("SHAP Explanation (Waterfall Plot)")

    # 使用 Matplotlib 设置图像分辨率为 300 DPI
    fig = plt.figure(dpi=300)
    shap.waterfall_plot(shap_values[1][0])  # 1表示预测为抗药性（Resistant）类别的SHAP值
    st.pyplot(fig)


    # 使用 SHAP 的 TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 打印 shap_values 结构以查看它的内容
    st.write("SHAP values structure: ", type(shap_values), shap_values)

    # 对于二分类问题，shap_values 可能是一个长度为2的列表
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_class_1 = shap_values[1]  # 获取类别1的 SHAP 值
        st.write("SHAP values for class 1: ", shap_values_class_1)
        
        # 生成 SHAP 条形图
        shap.plots.bar(shap_values_class_1)

        # 设置图像为 300 DPI
        plt.savefig("shap_bar_plot.png", dpi=300, bbox_inches='tight')

        # 在 Streamlit 中显示图像
        st.image("shap_bar_plot.png")

    else:
        st.write("SHAP values is not as expected. It may not be a list with 2 elements.")
