"""
Демо-дашборд мониторинга полевых работ
Исследование рынка жевательной резинки (аудит 500 ТТ + опрос 750 покупателей)

ВСЕ ДАННЫЕ ВЫДУМАНЫ и генерируются случайно (seed фиксирован).
Запуск:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ стиль
NAVY, BLUE, LIGHT, MINT, AMBER, GREY, RED = "#00287A", "#0067C5", "#7CC6EE", "#0A9B74", "#F2A900", "#8A99B3", "#D9363E"
STATUS_ORDER = ["Принято", "На проверке", "На исправлении", "Запланировано"]
STATUS_COLORS = {"Принято": MINT, "На проверке": BLUE, "На исправлении": AMBER, "Запланировано": GREY}
STATUS_RGB = {"Принято": [10, 155, 116, 230], "На проверке": [0, 103, 197, 230], "На исправлении": [242, 169, 0, 240], "Запланировано": [138, 153, 179, 200]}
CITY_CENTER = {"Душанбе": (38.5598, 68.7870), "Худжанд": (40.2826, 69.6220), "Бохтар": (37.8364, 68.7803)}
CITY_TT = {"Душанбе": 315, "Худжанд": 125, "Бохтар": 60}
CITY_AUDITORS = {"Душанбе": ["Аудитор 1", "Аудитор 2", "Аудитор 3", "Аудитор 4", "Аудитор 5"],
                 "Худжанд": ["Аудитор 6", "Аудитор 7"], "Бохтар": ["Аудитор 8"]}
CITY_PREFIX = {"Душанбе": "DU", "Худжанд": "KH", "Бохтар": "BO"}
AUDIT_DAYS = [8, 9, 10, 11, 12]
SURVEY_DAYS = list(range(13, 21))
ISSUES = ["Неразборчивое фото", "Цена ≠ ценник на фото", "GPS вне адреса ТТ", "Не заполнены SKU", "Дубль записи"]

st.set_page_config(page_title="Мониторинг полевых работ · демо", page_icon="📊", layout="wide")
st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.6rem; padding-bottom: 1rem;}}
      h1, h2, h3 {{color: {NAVY};}}
      div[data-testid="stMetric"] {{background: #EAF3FB; border-radius: 14px; padding: 14px 18px;}}
      div[data-testid="stMetricValue"] {{color: {NAVY}; font-weight: 700;}}
      .demo-badge {{display:inline-block; background:{AMBER}; color:#13213C; font-weight:700;
                   padding:3px 12px; border-radius:20px; font-size:0.8rem; margin-left:10px; vertical-align:middle;}}
      .sub {{color:#6B7790; margin-top:-8px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ данные
@st.cache_data
def make_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for city, n in CITY_TT.items():
        lat0, lon0 = CITY_CENTER[city]
        spread = {"Душанбе": 0.045, "Худжанд": 0.03, "Бохтар": 0.02}[city]
        auds = CITY_AUDITORS[city]
        fmt = rng.choice(["KA", "Non-KA", "BTC", "ОПТ"], size=n, p=[0.12, 0.38, 0.42, 0.08])
        # кластеры: каждому аудитору — свой сектор города, каждый день — свой кластер
        for i in range(n):
            aud = auds[i % len(auds)]
            a_idx = auds.index(aud)
            k = i // len(auds)
            per_aud = int(np.ceil(n / len(auds)))
            day = AUDIT_DAYS[min(int(k / (per_aud / 5)), 4)]
            ang = 2 * np.pi * (a_idx + (AUDIT_DAYS.index(day) + 0.5) / 5) / len(auds)
            r = spread * (0.35 + 0.55 * rng.random())
            lat = lat0 + r * np.sin(ang) + rng.normal(0, spread * 0.12)
            lon = lon0 + r * 1.25 * np.cos(ang) + rng.normal(0, spread * 0.15)
            rows.append(dict(tt=f"{CITY_PREFIX[city]}-{len([x for x in rows if x['city']==city])+1:04d}",
                             city=city, fmt=fmt[i], auditor=aud, day=day, lat=lat, lon=lon,
                             hour=int(rng.integers(9, 19)), minute=int(rng.integers(0, 60)),
                             u_issue=rng.random(), u_check=rng.random(), u_fix=rng.random(),
                             issue=rng.choice(ISSUES, p=[0.32, 0.26, 0.16, 0.16, 0.10]),
                             orbit=rng.random() < {"KA": 0.95, "Non-KA": 0.82, "BTC": 0.66, "ОПТ": 0.74}[fmt[i]],
                             dirol=rng.random() < {"KA": 0.93, "Non-KA": 0.85, "BTC": 0.72, "ОПТ": 0.80}[fmt[i]],
                             mentos=rng.random() < {"KA": 0.88, "Non-KA": 0.61, "BTC": 0.44, "ОПТ": 0.58}[fmt[i]],
                             photos=int(rng.integers(3, 8))))
    tt = pd.DataFrame(rows)

    # 50 ТТ для опроса: пропорционально городам
    survey_n = {"Душанбе": 31, "Худжанд": 13, "Бохтар": 6}
    picks = []
    for city, n in survey_n.items():
        picks += list(tt[tt.city == city].sample(n, random_state=seed).tt)
    tt["survey_point"] = tt.tt.isin(picks)

    # 750 интервью
    iv = []
    sp = tt[tt.survey_point].reset_index(drop=True)
    for _, p in sp.iterrows():
        n = 15
        days = rng.choice(SURVEY_DAYS, size=n)
        for d in days:
            iv.append(dict(tt=p.tt, city=p.city, fmt=p.fmt, day=int(d),
                           interviewer=p.auditor,
                           gender=rng.choice(["Мужчины", "Женщины"], p=[0.52, 0.48]),
                           age=rng.choice(["18–24", "25–34", "35–44", "45+"], p=[0.27, 0.33, 0.22, 0.18]),
                           duration=float(np.clip(rng.normal(8.4, 1.4), 4.5, 14)),
                           u_check=rng.random()))
    iv = pd.DataFrame(iv)
    return tt, iv


def status_on(tt: pd.DataFrame, today: int) -> pd.Series:
    """Статус каждой ТТ на конец выбранного дня проекта."""
    s = pd.Series("Запланировано", index=tt.index)
    visited = tt.day <= today
    same_day = tt.day == today
    prev_day = tt.day == today - 1
    older = tt.day < today - 1
    s[visited] = "Принято"
    s[same_day & (tt.u_check < 0.55)] = "На проверке"
    s[same_day & (tt.u_issue < 0.12)] = "На исправлении"
    s[prev_day & (tt.u_issue < 0.12) & (tt.u_fix < 0.5)] = "На исправлении"
    s[older & (tt.u_issue < 0.12) & (tt.u_fix < 0.08)] = "На исправлении"
    return s


tt, iv = make_data()

# ------------------------------------------------------------------ шапка + фильтры
st.markdown(
    f"<h2 style='margin-bottom:0'>Мониторинг полевых работ: рынок жевательной резинки"
    f"<span class='demo-badge'>ДЕМО-ДАННЫЕ</span></h2>",
    unsafe_allow_html=True,
)
st.markdown("<p class='sub'>Аудит 500 торговых точек и опрос 750 покупателей · Душанбе, Худжанд, Бохтар</p>",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### Фильтры")
    today = st.slider("День проекта", min_value=7, max_value=21, value=10,
                      help="Аудит: дни 8–12 · Опрос: дни 13–20")
    cities = st.multiselect("Город", list(CITY_TT), default=list(CITY_TT))
    fmts = st.multiselect("Формат ТТ", ["KA", "Non-KA", "BTC", "ОПТ"], default=["KA", "Non-KA", "BTC", "ОПТ"])
    auds = [a for c in cities for a in CITY_AUDITORS[c]]
    aud_sel = st.multiselect("Аудитор / интервьюер", auds, default=auds)
    st.divider()
    st.caption("Данные обновляются ежедневно по мере проверки записей офисной командой.")
    st.caption("⚠️ Все данные на этой странице выдуманы для демонстрации интерфейса.")

tt["status"] = pd.Categorical(status_on(tt, today), categories=STATUS_ORDER, ordered=True)
f = tt[tt.city.isin(cities) & tt.fmt.isin(fmts) & tt.auditor.isin(aud_sel)].copy()
fiv = iv[iv.city.isin(cities) & iv.fmt.isin(fmts) & iv.interviewer.isin(aud_sel)].copy()
fiv_done = fiv[fiv.day <= today]
plan_tt = len(f)
plan_iv = len(fiv)

phase = ("Подготовка" if today < 8 else "Аудит торговых точек" if today <= 12
         else "Опрос покупателей" if today <= 20 else "Обработка и анализ")
st.markdown(f"**День {today} из 30** · текущий этап: **{phase}**")

# ------------------------------------------------------------------ KPI
visited = int((f.status != "Запланировано").sum())
accepted = int((f.status == "Принято").sum())
on_check = int((f.status == "На проверке").sum())
on_fix = int((f.status == "На исправлении").sum())
import inspect
_ARROW = "delta_arrow" in inspect.signature(st.metric).parameters
def metric(col, label, value, delta=None):
    kw = {"delta_color": "off"}
    if _ARROW:
        kw["delta_arrow"] = "off"
    col.metric(label, value, delta, **kw)

k = st.columns(5)
metric(k[0], "Посещено ТТ", f"{visited} / {plan_tt}", f"{visited / max(plan_tt, 1):.0%} плана")
metric(k[1], "Принято офисом", accepted, f"{accepted / max(visited, 1):.0%} посещённых")
metric(k[2], "На проверке", on_check, "ждут проверки офисом")
metric(k[3], "На исправлении", on_fix, "возвращены аудиторам")
metric(k[4], "Интервью", f"{len(fiv_done)} / {plan_iv}", f"{len(fiv_done) / max(plan_iv, 1):.0%} плана")

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Карта и прогресс", "🧾 Аудиторы и контроль", "🗣️ Опрос покупателей", "🔍 Торговая точка"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    c1, c2 = st.columns([1.35, 1])
    with c1:
        city_for_map = st.radio("Карта", cities if cities else list(CITY_TT), horizontal=True, label_visibility="collapsed")
        m = f[f.city == city_for_map]
        m = m.assign(color=m.status.astype(str).map(STATUS_RGB), status_s=m.status.astype(str))
        legend = " &nbsp; ".join(f"<span style='color:{STATUS_COLORS[k]};font-size:1.1rem'>●</span> {k}" for k in STATUS_ORDER)
        st.markdown(f"<div style='font-size:0.85rem;margin:-6px 0 6px 0'>{legend}</div>", unsafe_allow_html=True)
        lat0, lon0 = CITY_CENTER[city_for_map]
        layer = pdk.Layer("ScatterplotLayer", data=m, get_position="[lon, lat]", get_fill_color="color",
                          get_line_color=[255, 255, 255], line_width_min_pixels=1.5, stroked=True,
                          get_radius=60, radius_min_pixels=6, radius_max_pixels=14, pickable=True)
        deck = pdk.Deck(layers=[layer], map_style="light",
                        initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0,
                                                         zoom=11.6 if city_for_map == "Душанбе" else 12.4),
                        tooltip={"html": "<b>{tt}</b><br/>{status_s}<br/>{fmt} · {auditor}<br/>день визита: {day}",
                                 "style": {"backgroundColor": "#00287A", "color": "white", "fontSize": "12px"}})
        st.pydeck_chart(deck, height=470)
    with c2:
        st.markdown("##### Прогресс по городам")
        g = (f.groupby(["city", "status"], observed=False).size().reset_index(name="n"))
        g = g[g.city.isin(cities)]
        fig2 = px.bar(g, y="city", x="n", color="status", orientation="h", color_discrete_map=STATUS_COLORS,
                      category_orders={"status": STATUS_ORDER, "city": list(CITY_TT)}, text="n", height=230)
        fig2.update_traces(textposition="inside", insidetextanchor="middle")
        fig2.update_layout(margin=dict(l=0, r=0, t=5, b=0), legend=dict(orientation="h", y=-0.25, title=None),
                           xaxis_title=None, yaxis_title=None, plot_bgcolor="white")
        st.plotly_chart(fig2, width="stretch")

        st.markdown("##### Динамика посещений, накопительно")
        days = list(range(7, 22))
        cum = pd.DataFrame({"День": days,
                            "План": [int((f.day <= d).sum()) for d in days],
                            "Факт": [int((f.day <= min(d, today)).sum()) if d <= today else None for d in days]})
        fig3 = go.Figure()
        fig3.add_scatter(x=cum["День"], y=cum["План"], name="План", line=dict(color=GREY, dash="dash"))
        fig3.add_scatter(x=cum["День"], y=cum["Факт"], name="Факт", line=dict(color=BLUE, width=3), mode="lines+markers")
        fig3.update_layout(height=210, margin=dict(l=0, r=0, t=5, b=0), plot_bgcolor="white",
                           legend=dict(orientation="h", y=1.15), xaxis=dict(dtick=1, title="день проекта"))
        st.plotly_chart(fig3, width="stretch")

# ------------------------------------------------------------------ TAB 2
with tab2:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("##### Выполнение по аудиторам")
        a = f.groupby(["city", "auditor"]).agg(
            План=("tt", "size"),
            Посещено=("status", lambda s: int((s != "Запланировано").sum())),
            Принято=("status", lambda s: int((s == "Принято").sum())),
            На_исправлении=("status", lambda s: int((s == "На исправлении").sum())),
        ).reset_index().rename(columns={"city": "Город", "auditor": "Аудитор", "На_исправлении": "На исправлении"})
        a["Выполнение"] = (100 * a["Посещено"] / a["План"]).round(0)
        st.dataframe(a, hide_index=True, width="stretch", height=330,
                     column_config={"Выполнение": st.column_config.ProgressColumn("Выполнение", format="%.0f%%",
                                                                                  min_value=0, max_value=100)})
    with c2:
        st.markdown("##### Причины возврата на исправление")
        fx = f[f.status == "На исправлении"]
        if len(fx):
            ic = fx.issue.value_counts().reindex(ISSUES).fillna(0).reset_index()
            ic.columns = ["Причина", "ТТ"]
            fig4 = px.bar(ic, x="ТТ", y="Причина", orientation="h", text="ТТ", color_discrete_sequence=[AMBER], height=300)
            fig4.update_layout(margin=dict(l=0, r=0, t=5, b=0), plot_bgcolor="white", yaxis=dict(autorange="reversed", title=None))
            st.plotly_chart(fig4, width="stretch")
        else:
            st.info("Нет записей на исправлении на выбранный день.")
    st.markdown("##### Записи, возвращённые на исправление")
    lst = f[f.status == "На исправлении"][["tt", "city", "fmt", "auditor", "day", "issue"]].rename(
        columns={"tt": "Код ТТ", "city": "Город", "fmt": "Формат", "auditor": "Аудитор", "day": "День визита", "issue": "Причина"})
    lst["Действие"] = lst["Причина"].map({"Неразборчивое фото": "Запросить повторное фото",
                                          "Цена ≠ ценник на фото": "Перепроверить запись",
                                          "GPS вне адреса ТТ": "Уточнить / повторный визит",
                                          "Не заполнены SKU": "Дозаполнить форму",
                                          "Дубль записи": "Сверить и удалить дубль"})
    st.dataframe(lst, hide_index=True, width="stretch", height=220)

# ------------------------------------------------------------------ TAB 3
with tab3:
    if today < 13:
        st.info(f"Опрос покупателей начнётся на 13-й день проекта (сейчас день {today}). "
                "Ниже — план распределения интервью.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Интервью по городам: план и факт")
        pc = fiv.groupby("city").size().rename("План").to_frame()
        pc["Факт"] = fiv_done.groupby("city").size()
        pc = pc.fillna(0).reindex([c for c in CITY_TT if c in pc.index]).reset_index().melt(id_vars="city", var_name="", value_name="n")
        fig5 = px.bar(pc, x="city", y="n", color="", barmode="group", text="n",
                      color_discrete_map={"План": "#C9D6E8", "Факт": BLUE}, height=300)
        fig5.update_layout(margin=dict(l=0, r=0, t=5, b=0), plot_bgcolor="white", xaxis_title=None, yaxis_title=None,
                           legend=dict(orientation="h", y=1.12, title=None))
        st.plotly_chart(fig5, width="stretch")
    with c2:
        st.markdown("##### Выполнение половозрастных квот")
        q = fiv.groupby(["gender", "age"]).size().rename("План").to_frame()
        q["Факт"] = fiv_done.groupby(["gender", "age"]).size()
        q = q.fillna(0)
        q["pct"] = q["Факт"] / q["План"]
        hm = q["pct"].unstack("age")[["18–24", "25–34", "35–44", "45+"]]
        fig6 = px.imshow(hm, text_auto=".0%", color_continuous_scale=["#EAF3FB", LIGHT, BLUE, NAVY], zmin=0, zmax=1,
                         aspect="auto", height=300)
        fig6.update_layout(margin=dict(l=0, r=0, t=5, b=0), coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig6, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Интервью по дням")
        dd = fiv.groupby("day").size().rename("План").to_frame()
        dd["Факт"] = fiv_done.groupby("day").size()
        dd = dd.reset_index()
        fig7 = go.Figure()
        fig7.add_bar(x=dd["day"], y=dd["План"], name="План", marker_color="#C9D6E8")
        fig7.add_bar(x=dd["day"], y=dd["Факт"], name="Факт", marker_color=MINT)
        fig7.update_layout(barmode="overlay", height=250, margin=dict(l=0, r=0, t=5, b=0), plot_bgcolor="white",
                           xaxis=dict(dtick=1, title="день проекта"), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig7, width="stretch")
    with c4:
        st.markdown("##### Контроль длительности интервью")
        if len(fiv_done):
            fig8 = px.histogram(fiv_done, x="duration", nbins=24, color_discrete_sequence=[BLUE], height=250)
            fig8.add_vrect(x0=0, x1=5, fillcolor=RED, opacity=0.12, line_width=0, annotation_text="< 5 мин — проверка",
                           annotation_position="top left")
            fig8.update_layout(margin=dict(l=0, r=0, t=5, b=0), plot_bgcolor="white", xaxis_title="минут", yaxis_title=None)
            st.plotly_chart(fig8, width="stretch")
            short = int((fiv_done.duration < 5).sum())
            st.caption(f"Средняя длительность: {fiv_done.duration.mean():.1f} мин · коротких интервью на проверке: {short}")
        else:
            st.caption("Данные появятся после начала опроса.")

# ------------------------------------------------------------------ TAB 4
with tab4:
    done = f[f.status != "Запланировано"]
    if len(done) == 0:
        st.info("На выбранный день посещённых точек нет.")
    else:
        code = st.selectbox("Код торговой точки", done.tt.tolist())
        p = done[done.tt == code].iloc[0]
        c1, c2 = st.columns([1, 1.3])
        with c1:
            color = STATUS_COLORS[p.status]
            st.markdown(f"<div style='background:{color};color:white;padding:8px 14px;border-radius:20px;"
                        f"display:inline-block;font-weight:700'>{p.status}</div>", unsafe_allow_html=True)
            st.markdown(f"""
| Поле | Значение |
|---|---|
| Город / формат | {p.city} · {p.fmt} |
| Аудитор | {p.auditor} |
| Визит | день {p.day}, {p.hour:02d}:{p.minute:02d} |
| GPS | {p.lat:.4f}, {p.lon:.4f} |
| Orbit / Dirol / Mentos | {"✓" if p.orbit else "✗"} / {"✓" if p.dirol else "✗"} / {"✓" if p.mentos else "✗"} |
| Фото | {p.photos} шт. |
| Точка опроса | {"да" if p.survey_point else "нет"} |
""")
            if p.status == "На исправлении":
                st.warning(f"Возвращено: {p.issue}")
        with c2:
            st.markdown("##### Фотоотчёт")
            cols = st.columns(3)
            labels = ["Общая выкладка", "Прикассовая зона", "Ценники"]
            for i, col in enumerate(cols):
                col.markdown(
                    f"<div style='background:#EAF3FB;border-radius:12px;height:150px;display:flex;flex-direction:column;align-items:center;"
                    f"justify-content:center;color:{GREY};font-size:0.85rem;text-align:center'><div style='font-size:1.6rem'>📷</div><div>{labels[i]}</div>"
                    f"<div style='font-size:0.7rem'>демо-заглушка</div></div>", unsafe_allow_html=True)

st.caption("Демонстрационный интерфейс. Все цифры, коды точек и координаты сгенерированы случайно и не являются результатами исследования.")
