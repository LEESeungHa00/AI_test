import streamlit as st
import pandas as pd
import numpy as np
import time


# --- 1. 기본 설정 및 스타일 ---
st.set_page_config(page_title="Trade Insight AI", page_icon="🧀", layout="centered")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: 1px solid #d1d5db;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 15px;
    }
    .main-header {
        text-align: center;
        margin-bottom: 0.5rem;
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 세션 상태 초기화 ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# --- 3. 데이터 로드 ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mozzarella_data.csv")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        # dummy
        dates = pd.date_range(start='2023-01-01', periods=200)
        countries = ['USA', 'Germany', 'New Zealand', 'Denmark', 'Italy']
        data = {
            'Date': dates,
            'HS Code': ['0406.10'] * 200,
            'Product Name': ['Mozzarella Shredded'] * 100 + ['Mozzarella Block'] * 100,
            'Origin Country': np.random.choice(countries, 200, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            'Volume': np.random.randint(5, 50, 200),
            'Unit Price': np.random.uniform(4.5, 7.5, 200)
        }
        return pd.DataFrame(data)

df = load_data()

# --- 4. 프롬프트 엔지니어링 모듈 (Prompt Builder - Professional Ver.) ---
class TradePromptBuilder:
    """
    Refined Prompt Engineering for B2B Professional Context
    """
    @staticmethod
    def build_system_prompt():
        return """
        # ROLE (Persona)
        You are a Strategic Sourcing Consultant for a Fortune 500 company.
        Your goal is to provide objective, data-driven insights to help procurement managers optimize costs and manage risks.
        
        # TONE & MANNER
        - Professional, Objective, and Constructive.
        - Avoid slang, aggressive language, or blame (e.g., avoid "rip-off", "sucker").
        - Use business terminology (e.g., "Cost Optimization", "Market Positioning", "Leverage").
        - Focus on financial impact and strategic opportunities.
        """

    @staticmethod
    def build_user_prompt(q_type, market_data, user_data):
        # 1. Chain-of-Thought (CoT)
        cot_instruction = """
        # THINKING PROCESS
        1. Compare User Price vs. Market Fair Price (adjusted for volume).
        2. Calculate the Price Variance (%) and Potential Annual Savings ($).
        3. Assess the user's market position (Competitive vs. Needs Improvement).
        4. Suggest strategic next steps (e.g., supplier diversification, renegotiation).
        """
        
        # 2. Directional Stimulus (B2B Keywords)
        stimulus = """
        # KEYWORDS TO USE
        - 'Market Variance' (시장 격차)
        - 'Cost Efficiency' (비용 효율성)
        - 'Strategic Sourcing' (전략적 소싱)
        - 'Potential Savings' (절감 잠재력)
        """
        
        context = f"""
        # CONTEXT
        - Analysis: {q_type}
        - Market Avg: ${market_data['avg_price']:.2f}
        - User Price: ${user_data['my_price']:.2f}
        - User Volume: {user_data['my_volume']} tons
        """
        
        return f"{cot_instruction}\n{stimulus}\n{context}\n\nOutput JSON."

# --- 5. 분석 엔진 (Simulation Logic - Refined) ---
def run_llm_analysis(df, q_type, detailed_input):
    origin = detailed_input['target_origin']
    my_price = detailed_input['my_price']
    my_vol = detailed_input['my_volume']
    
    # Data Filtering
    target_df = df[df['Origin Country'] == origin]
    market_avg = target_df['Unit Price'].mean() if not target_df.empty else df['Unit Price'].mean()
    
    # Logic Simulation
    result = {}
    
    if "단가 적정성" in q_type:
        discount_factor = 0.95 if my_vol >= 20 else 1.0
        fair_price = market_avg * discount_factor
        gap_pct = ((my_price - fair_price) / fair_price) * 100
        loss = int((my_price - fair_price) * my_vol * 1000)
        
        # Status Definition
        result['status'] = "Needs Improvement" if gap_pct > 0 else "Competitive"
        
        # Professional Messaging
        if gap_pct > 0:
            result['title'] = "📉 비용 효율화 필요 (Cost Optimization Needed)"
            result['summary'] = f"현재 매입가는 시장 적정가(${fair_price:.2f}) 대비 **{gap_pct:.1f}% 상회**하고 있습니다. 이는 현재 물량 규모 대비 최적화된 조건이 아님을 시사합니다."
            result['impact'] = loss
            result['hook_msg'] = "동일 스펙 기준, **비용 절감이 가능한 대체 공급 국가 2곳**의 시장 데이터를 확보했습니다."
        else:
            result['title'] = "✅ 가격 경쟁력 우수 (Highly Competitive)"
            result['summary'] = "시장 상위 10% 수준의 우수한 단가로 매입 중입니다. 현재의 경쟁력을 유지하며 공급 안정성을 강화할 시점입니다."
            result['impact'] = 0
            result['hook_msg'] = "현 단가 수준을 유지하면서 **공급망 리스크를 분산할 수 있는 이중화 전략**을 제안합니다."
            
        result['chart_label'] = ["시장 적정가", "귀사 매입가"]
        result['chart_val'] = [fair_price, my_price]

    elif "대체 공급처" in q_type:
        country_avg = df.groupby('Origin Country')['Unit Price'].mean()
        cheaper = country_avg[country_avg < market_avg].index.tolist()
        
        result['status'] = "Opportunity"
        result['title'] = "💡 소싱 다변화 기회 (Sourcing Opportunity)"
        result['summary'] = f"현재 거래 중인 **{origin}** 대비, 가격 경쟁력이 우수한 국가가 **{len(cheaper)}곳** 식별되었습니다. 전략적 소싱 전환을 검토해 보십시오."
        result['impact'] = None
        result['chart_label'] = [origin, "대체 국가 평균"]
        result['chart_val'] = [market_avg, country_avg[cheaper].mean() if cheaper else market_avg]
        result['hook_msg'] = f"**{cheaper[0] if cheaper else 'New Origin'}** 내 검증된 우량 공급사(Top-tier Suppliers) 리스트를 확인하시겠습니까?"
        
    else:
        result['status'] = "Info"
        result['title'] = "📊 분석 완료"
        result['summary'] = "요청하신 데이터에 대한 정밀 분석이 완료되었습니다. 상세 지표를 확인해 주십시오."
        result['impact'] = None
        result['chart_label'] = ["시장 평균", "귀사 타겟"]
        result['chart_val'] = [market_avg, my_price]
        result['hook_msg'] = "상세 분석 리포트 및 원본 데이터를 다운로드하시겠습니까?"

    return result

# --- 6. UI 렌더링 ---

st.markdown("<h1 class='main-header'>🧀 Trade Insight AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Global Trade Intelligence for Strategic Sourcing</p>", unsafe_allow_html=True)

# [Step 1] Init
if st.session_state.step == 1:
    st.info("💡 **Insight:** HS Code를 입력하시면 AI가 전 세계 시장 데이터를 분석하여 Global Market Brief를 제공합니다.")
    
    with st.form("init_form"):
        col1, col2 = st.columns(2)
        hscode = col1.text_input("HS Code", value="0406.10")
        product_name = col2.text_input("품목명 (Product Name)", value="Mozzarella Cheese")
        
        col3, col4 = st.columns(2)
        target_country = col3.selectbox("관심 국가 (Optional)", ["선택 안함"] + list(df['Origin Country'].unique()))
        exclude = col4.text_input("제외 국가 (Optional)", placeholder="Ex: China")
        
        submitted = st.form_submit_button("🚀 시장 데이터 스캔 (Scan Market)")
        
        if submitted:
            st.session_state.user_data.update({
                'hscode': hscode, 'product': product_name,
                'target_origin': target_country if target_country != "선택 안함" else df['Origin Country'].mode()[0],
                'exclude': exclude
            })
            st.session_state.step = 2
            st.rerun()

# [Step 2] Overview
elif st.session_state.step == 2:
    data = st.session_state.user_data
    avg_p = df['Unit Price'].mean()
    top_o = df.groupby('Origin Country')['Volume'].sum().idxmax()
    
    st.subheader(f"📊 Market Brief: {data['product']}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Global Avg Price", f"${avg_p:.2f}/kg")
    c2.metric("Dominant Origin", top_o)
    c3.metric("Market Trend", "Rising 🔼")
    
    st.markdown(f"""
    <div class='metric-card'>
    <b>🤖 Strategic Insight:</b><br>
    현재 시장은 <b>{top_o}</b> 공급 물량이 주도하고 있으며, 가격 변동성이 확대되는 추세입니다. 
    최적의 의사결정을 위해 <b>실제 거래 조건(물량, 단가)</b>에 기반한 정밀 포지셔닝 진단을 권장합니다.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("👉 정밀 진단 시작하기 (Start Deep-Dive)"):
        st.session_state.step = 3
        st.rerun()

# [Step 3] Type Select
elif st.session_state.step == 3:
    st.subheader("2️⃣ 분석 목적을 선택해 주세요.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚢 수입 최적화 (Import Optimization)"):
            st.session_state.mode = "Import"
            st.session_state.step = 4
            st.rerun()
    with c2:
        if st.button("✈️ 수출 확장 (Export Expansion)"):
            st.session_state.mode = "Export"
            st.session_state.step = 4
            st.rerun()

# [Step 4] Question Select
elif st.session_state.step == 4:
    st.subheader(f"3️⃣ {st.session_state.mode} 관련 핵심 이슈 선택")
    
    if st.session_state.mode == "Import":
        options = [
            "💰 단가 적정성 진단 (Price Competitiveness)", 
            "🌍 대체 공급처 발굴 (Sourcing Diversification)",
            "⏱️ 최적 구매 타이밍 (Market Timing)", 
            "🕵️ 경쟁사 동향 분석 (Competitor Intelligence)",
            "🚨 공급망 리스크 점검 (Supply Chain Risk)", 
            "🥩 스펙별 가성비 비교 (Spec Analysis)"
        ]
    else:
        options = [
            "💎 고마진 국가 탐색 (High-Margin Markets)", 
            "🚩 블루오션 발굴 (Blue Ocean Strategy)",
            "👑 진성 바이어 추출 (Key Buyer Identification)", 
            "💔 이탈 징후 감지 (Churn Risk)",
            "⚔️ 경쟁 강도 분석 (Market Share)", 
            "🚀 급성장 시장 예측 (Growth Opportunities)"
        ]
        
    choice = st.radio("분석 항목:", options)
    
    if st.button("다음 단계로 (Next)"):
        st.session_state.question = choice
        st.session_state.step = 5
        st.rerun()

# [Step 5] Detail Input
elif st.session_state.step == 5:
    st.subheader("4️⃣ 정밀 진단을 위한 데이터 입력")
    st.markdown(f"**'{st.session_state.question}'** 분석을 위해 구체적인 거래 조건을 입력해 주세요.")
    
    with st.form("detail_form"):
        origins = list(df['Origin Country'].unique())
        default_idx = origins.index(st.session_state.user_data['target_origin']) if st.session_state.user_data['target_origin'] in origins else 0
        target = st.selectbox("분석 대상 국가", origins, index=default_idx)
        
        c1, c2 = st.columns(2)
        vol = c1.number_input("연간 거래 물량 (Tons)", value=10.0, step=1.0)
        price = c2.number_input("매입(견적) 단가 ($/kg)", value=6.5, step=0.1)
        
        submit = st.form_submit_button("🔥 AI 진단 실행 (Run Analysis)")
        
        if submit:
            st.session_state.user_data.update({
                'target_origin': target,
                'my_volume': vol,
                'my_price': price
            })
            st.session_state.step = 6
            st.rerun()

# [Step 6] Result
elif st.session_state.step == 6:
    with st.spinner("Analyzing market data..."):
        time.sleep(1.0)
        res = run_llm_analysis(df, st.session_state.question, st.session_state.user_data)
    
    st.subheader("✅ AI Diagnostic Report")
    
    # Insight Box
    if "Needed" in res.get('status', ''):
        st.error(f"### {res['title']}")
    elif "Opportunity" in res.get('status', ''):
        st.success(f"### {res['title']}")
    else:
        st.info(f"### {res['title']}")
        
    st.markdown(f"**Insight:** {res['summary']}")
    
    if res['impact']:
        loss_krw = int(res['impact'] * 1300)
        st.markdown(f"📉 **예상 절감 기회 (Potential Savings): ${res['impact']:,} (약 {loss_krw//10000:,}만 원)**")
    
    st.divider()
    
    # Chart
    st.markdown("**📊 Positioning Chart**")
    chart_df = pd.DataFrame({
        "Category": res['chart_label'],
        "Price ($)": res['chart_val']
    })
    st.bar_chart(chart_df.set_index("Category"))
    
    # Hook
    st.warning("🔒 **Premium Insight (Locked)**")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"""
        * **{res['hook_msg']}**
        * Top 3 Recommended Suppliers: S******, M****, K********
        * Target Negotiation Price: $5.** / kg
        """)
    with c2:
        st.button("🔓 잠금 해제\n(Request Demo)", type="primary")
    
    st.markdown("---")
    if st.button("🔄 새로운 분석 시작하기 (Restart)"):
        st.session_state.step = 1
        st.rerun()
