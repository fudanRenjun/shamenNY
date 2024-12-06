import streamlit as st
import joblib
import numpy as np
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

    # 计算 SHAP 值，返回的是 Explanation 对象
    shap_values = explainer.shap_values(features)

    # 显示 SHAP waterfall 图
    st.subheader("SHAP Explanation (Waterfall Plot)")

    # 确保返回的 shap_values 是 Explanation 对象
    if isinstance(shap_values, list):
        # 如果返回多个类别的 SHAP 值，选择一个类别的解释对象
        shap_value = shap_values[0]  # 选择类别 0 (易感性) 或类别 1 (抗药性)
    else:
        # 如果只有一个类别的 SHAP 值，直接选择该解释对象
        shap_value = shap_values
    
    # `shap_value` 必须是 Explanation 对象，确保类型正确
    if isinstance(shap_value, list):
        shap_value = shap_value[0]  # 获取第一个样本的解释对象

    # 传递给 waterfall_plot 的应该是 Explanation 对象
    shap.waterfall_plot(shap_value)

    # 使用 Matplotlib 设置图像分辨率为 300 DPI
    fig = plt.figure(dpi=300)
    st.pyplot(fig)

