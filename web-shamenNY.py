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
input_13637 = st.number_input("13637:", min_value=0.000000001, max_value=1.0, value=0.00011864, format="%.9f")
input_9869 = st.number_input("9869±1:", min_value=0.0000000001, max_value=1.0, value=0.000149248, format="%.9f")

# 将输入的数据转化为模型的输入格式
feature_values = [
    input_2965, input_6567, input_11641, input_3929, input_4767, input_13637, input_9869
]
features = np.array([feature_values])

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

    # 使用 SHAP 的通用解释器
    explainer = shap.Explainer(model, pd.DataFrame([feature_values], columns=feature_names))
    shap_values = explainer(features)

    # 打印 SHAP 结果和 expected_value
    st.write("SHAP Values: ", shap_values)
    st.write("Expected Value: ", explainer.expected_value)

    # 获取 SHAP 值数组
    shap_values_array = shap_values.values

    # 打印 SHAP 值的结构
    st.write("SHAP values array structure: ", shap_values_array.shape)

    # 根据预测类别生成并显示 SHAP waterfall plot
    if predicted_class == 1:
        st.write("Generating Waterfall Plot for class 1 (Resistant)")
        shap.plots.waterfall(shap_values[1])  # 生成类别1的 Waterfall Plot
    else:
        st.write("Generating Waterfall Plot for class 0 (Susceptible)")
        shap.plots.waterfall(shap_values[0])  # 生成类别0的 Waterfall Plot

    # 保存SHAP图并显示
    plt.tight_layout()
    st.pyplot(plt)  # 用 Streamlit 显示交互式图形而不是保存到文件

