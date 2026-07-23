import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 页面基础配置
# ==========================================
st.set_page_config(page_title="HxAim 准星解析器", page_icon="🎯", layout="centered")

st.title("🎯 HxAim 准星 HSV 智能提取工具")
st.markdown("""
把你在游戏内按 `F11` 截取的图片（例如 `hxaim_crosshair_capture.jpg`）上传到这里，
工具会自动提取正中心的高亮准星，并转换成底层的 C++ OpenCV 数组。
""")

# ==========================================
# 侧边栏：高级参数调节
# ==========================================
st.sidebar.header("⚙️ 算法调优参数")
st.sidebar.markdown("一般保持默认即可，如遇提取失败可适当放宽。")

box_size = st.sidebar.slider("中心采样矩阵大小 (像素)", min_value=3, max_value=50, value=20, step=1)
min_s = st.sidebar.slider("剔除背景 - 最低饱和度 (S)", min_value=0, max_value=255, value=15)
min_v = st.sidebar.slider("剔除背景 - 最低明度 (V)", min_value=0, max_value=255, value=15)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 容错扩展 (Margin)")
h_margin = st.sidebar.slider("色调 H 上下浮动", 0, 30, 8)
s_margin = st.sidebar.slider("饱和度 S 上下浮动", 0, 100, 30)
v_margin = st.sidebar.slider("明度 V 上下浮动", 0, 100, 30)

# ==========================================
# 核心处理逻辑
# ==========================================
uploaded_file = st.file_uploader("📂 上传游戏截图", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 将上传的文件转为 OpenCV 的 BGR 格式矩阵
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ 无法解析图像，请检查文件是否损坏！")
    else:
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # 计算裁剪区域
        x = max(0, cx - box_size // 2)
        y = max(0, cy - box_size // 2)
        crop_w = min(w - x, box_size)
        crop_h = min(h - y, box_size)

        roi = img[y:y+crop_h, x:x+crop_w]

        # 图像展示区
        st.markdown("### 🖼️ 画面预览")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
        with col2:
            # 使用临近插值法放大中心区域，防止边缘模糊，方便观察像素点
            roi_resized = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_NEAREST)
            st.image(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB), caption=f"中心提取区 ({box_size}x{box_size})", use_container_width=False)

        # ----------------------------------------
        # 核心算法：提取 OpenCV 格式的 HSV
        # ----------------------------------------
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_list, s_list, v_list = [], [], []

        for r in range(hsv_roi.shape[0]):
            for c in range(hsv_roi.shape[1]):
                h_val, s_val, v_val = hsv_roi[r, c]
                # 过滤低饱和度、低亮度的背景墙壁颜色
                if s_val > min_s and v_val > min_v:
                    h_list.append(h_val)
                    s_list.append(s_val)
                    v_list.append(v_val)

        st.markdown("### 📊 提取结果")
        if not h_list:
            st.error("⚠️ 未在中心提取到彩色像素！原因可能是：\n1. 准星是纯白/黑色的（S和V过低）。\n2. 游戏截出来是纯黑的（全屏独占模式导致）。")
        else:
            # 计算原始极值
            # 计算原始极值，必须套上 int() 转换，防止 OpenCV uint8 溢出！
            raw_h_min, raw_h_max = int(min(h_list)), int(max(h_list))
            raw_s_min, raw_s_max = int(min(s_list)), int(max(s_list))
            raw_v_min, raw_v_max = int(min(v_list)), int(max(v_list))

            # 加入容错浮动
            final_h_min = max(0, raw_h_min - h_margin)
            final_h_max = min(180, raw_h_max + h_margin)
            final_s_min = max(0, raw_s_min - s_margin)
            final_s_max = min(255, raw_s_max + s_margin)
            final_v_min = max(0, raw_v_min - v_margin)
            final_v_max = min(255, raw_v_max + v_margin)

            # 数据指标卡片展示
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("色调 (H)", f"{int(final_h_min)} ~ {int(final_h_max)}")
            m_col2.metric("饱和度 (S)", f"{int(final_s_min)} ~ {int(final_s_max)}")
            m_col3.metric("明度 (V)", f"{int(final_v_min)} ~ {int(final_v_max)}")

            # 格式化 C++ 代码函数
            def format_cpp_array(h1, s1, v1, h2, s2, v2):
                return f"{{ {int(h1)}, {int(s1)}, {int(v1)}, 0, {int(h2)}, {int(s2)}, {int(v2)}, 0 }}"

            st.success("✅ 解析成功！请点击右上角复制下方代码，替换 DLL 中的旧数组：")

            # 跨色环红色处理
            if raw_h_min < 15 and raw_h_max > 165:
                st.warning("🚨 检测到红色跨越了 0° 与 180° 的色环边界，已自动切分为两组参数：")
                st.code(format_cpp_array(0, final_s_min, final_v_min, 10, final_s_max, final_v_max), language="cpp")
                st.code(format_cpp_array(156, final_s_min, final_v_min, 180, final_s_max, final_v_max), language="cpp")
            else:
                st.code(format_cpp_array(final_h_min, final_s_min, final_v_min, final_h_max, final_s_max, final_v_max), language="cpp")
