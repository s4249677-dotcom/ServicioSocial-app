import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="(EDA) - CADHAC", layout="wide")
st.title("Análisis Exploratorio de Datos (EDA) - CADHAC")
st.caption("La base de datos fue previamente diagnosticada y limpiada para su uso, en este tablero se muestra un análisis correspondiente a cada variable (características) de los individuos.")


@st.cache_data(show_spinner=False)
def get_df():
     df = pd.read_excel("Data_Cleaned_CADHAC.xlsx", sheet_name=0)
     return df  

COL_YEAR = "Año de recepción del caso"
COL_GENDER = "Víctima: Género"
COL_EDU = "Víctima: Escolaridad"
COL_OCC = "Víctima: Ocupación (Grupo según Encuesta Nacional de Ocupación y Empleo)"
COL_CIVIL = "Víctima: Estado civil"
COL_MUNI = "Víctima: Municipio de Residencia"
EXPECTED_COLS = [COL_YEAR, COL_GENDER, COL_EDU, COL_OCC, COL_CIVIL, COL_MUNI]

with st.sidebar:
    st.header("Opciones")
    include_unknown = st.checkbox("Incluir categoría 'Desconocido'", value=True)
    top_n_long = st.slider("Top N para listas largas (Municipio / Ocupación)", 5, 100, 30, step=5)

def ensure_year_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if COL_YEAR in df.columns:
        df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors='coerce')
        if df[COL_YEAR].notna().any():
            try:
                df[COL_YEAR] = df[COL_YEAR].astype('Int64')
            except Exception:
                pass
    return df

def standardize_gender(df: pd.DataFrame) -> pd.DataFrame:
    if COL_GENDER in df.columns:
        df[COL_GENDER] = (
            df[COL_GENDER]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.capitalize()
        )
        df.loc[df[COL_GENDER].isin(["nan", "none", "", "n/a", "na"]), COL_GENDER] = None
    return df

def fill_unknown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].where(~df[col].isna(), "Desconocido")
        df[col] = df[col].replace({"": "Desconocido", " ": "Desconocido"})
    return df

def counts_df(df: pd.DataFrame, col: str, include_unknown: bool, top_n: int | None = None) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "Frecuencia"])  # vacío
    series = df[col]
    if not include_unknown:
        series = series[series.astype(str).str.strip().str.lower() != "desconocido"]
        series = series.dropna()
    vc = series.value_counts(dropna=False)
    out = vc.rename_axis(col).reset_index(name="Frecuencia")
    if top_n is not None and len(out) > top_n:
        out = out.head(top_n)
    return out


def apply_topn(series: pd.Series, top_n: int, other_label: str = "Otros") -> pd.Series:
    counts = series.value_counts()
    keep = counts.head(top_n).index
    return series.where(series.isin(keep), other_label)


def filter_unknowns(df: pd.DataFrame, cols: list[str], include_unknown: bool) -> pd.DataFrame:
    if not include_unknown:
        for c in cols:
            if c in df.columns:
                df = df[df[c].astype(str).str.strip().str.lower() != "desconocido"]
    return df

df_raw = get_df()
st.success("Base de datos lista")
st.write("**Dimensiones:**", df_raw.shape)

df = df_raw.copy()
df = ensure_year_numeric(df)
df = standardize_gender(df)
for c in [COL_GENDER, COL_EDU, COL_OCC, COL_CIVIL, COL_MUNI]:
    df = fill_unknown(df, c)

if COL_YEAR in df.columns and df[COL_YEAR].notna().any():
    min_year = int(df[COL_YEAR].min())
    max_year = int(df[COL_YEAR].max())
    year_range = st.slider(
        "Rango de años",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
        help="Filtra todas las visualizaciones por año de recepción del caso"
    )
    df_filtered = df[(df[COL_YEAR] >= year_range[0]) & (df[COL_YEAR] <= year_range[1])]
else:
    df_filtered = df

tabs = st.tabs([
    "Casos por año",
    "Género",
    "Escolaridad",
    "Ocupación",
    "Estado civil",
    "Municipio",
    "Bivariado",
    "Pirámide Bivariado", 
    "Multivariado"
])

with tabs[0]:
    if COL_YEAR in df_filtered.columns and df_filtered[COL_YEAR].notna().any():
        fig_0 = px.histogram(
            df_filtered,
            x=COL_YEAR,
            color=COL_YEAR,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Casos por año con colores personalizados",
        )
        fig_0.update_xaxes(dtick=1)
        fig_0.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_0, use_container_width=True)
    else:
        st.warning(f"No se encontró la columna '{COL_YEAR}' o no tiene valores válidos.")

with tabs[1]:
    if COL_GENDER in df_filtered.columns:
        df_gen = df_filtered.copy()
        if not include_unknown:
            df_gen = df_gen[df_gen[COL_GENDER].astype(str).str.lower() != "desconocido"]
        fig_1 = px.pie(
            df_gen,
            names=COL_GENDER,
            title="Género"
        )
        st.plotly_chart(fig_1, use_container_width=True)
    else:
        st.warning(f"No se encontró la columna '{COL_GENDER}'.")

with tabs[2]:
    if COL_EDU in df_filtered.columns:
        counts = counts_df(df_filtered, COL_EDU, include_unknown)
        if len(counts):
            fig = px.bar(
                counts,
                x=COL_EDU,
                y="Frecuencia",
                title="Frecuencia de escolaridad",
                text="Frecuencia"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis={'categoryorder': 'total descending'}, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ver tabla"):
                st.dataframe(counts)
        else:
            st.info("No hay datos para mostrar.")
    else:
        st.warning(f"No se encontró la columna '{COL_EDU}'.")

with tabs[3]:
    if COL_OCC in df_filtered.columns:
        counts = counts_df(df_filtered, COL_OCC, include_unknown, top_n=top_n_long)
        if len(counts):
            counts = counts.rename(columns={COL_OCC: "Ocupación"})
            fig = px.bar(
                counts,
                x="Ocupación",
                y="Frecuencia",
                title="Ocupación",
                text="Frecuencia"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis={'categoryorder': 'total descending'}, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ver tabla"):
                st.dataframe(counts)
        else:
            st.info("No hay datos para mostrar.")
    else:
        st.warning(f"No se encontró la columna '{COL_OCC}'.")

with tabs[4]:
    if COL_CIVIL in df_filtered.columns:
        df_ec = df_filtered.copy()
        if not include_unknown:
            df_ec = df_ec[df_ec[COL_CIVIL].astype(str).str.lower() != "desconocido"]
        fig_ec = px.pie(
            df_ec,
            names=COL_CIVIL,
            title="Estado civil"
        )
        st.plotly_chart(fig_ec, use_container_width=True)
    else:
        st.warning(f"No se encontró la columna '{COL_CIVIL}'.")

with tabs[5]:
    if COL_MUNI in df_filtered.columns:
        counts = counts_df(df_filtered, COL_MUNI, include_unknown, top_n=top_n_long)
        if len(counts):
            counts = counts.rename(columns={COL_MUNI: "Municipio"})
            fig_2 = px.bar(
                counts,
                x="Municipio",
                y="Frecuencia",
                title="Municipio",
                text="Frecuencia"
            )
            fig_2.update_traces(textposition="outside")
            fig_2.update_layout(xaxis={'categoryorder': 'total descending'}, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_2, use_container_width=True)
            with st.expander("Ver tabla"):
                st.dataframe(counts)
        else:
            st.info("No hay datos para mostrar.")
    else:
        st.warning(f"No se encontró la columna '{COL_MUNI}'.")

with tabs[6]:
    st.subheader("Análisis bivariado")
    st.caption("Cruza dos variables categóricas o analiza una categoría a través del tiempo.")

    col_options = {
        "Año de recepción del caso (categoría)": COL_YEAR,
        "Género": COL_GENDER,
        "Escolaridad": COL_EDU,
        "Ocupación": COL_OCC,
        "Estado civil": COL_CIVIL,
        "Municipio": COL_MUNI,
    }

    chart_type = st.radio(
        "Tipo de análisis",
        [
            "Heatmap (X vs Y)",
            "Barras apiladas (conteo)",
            "Barras apiladas (%)",
            "Tendencia por año",
        ],
        index=0,
        horizontal=False,
    )

    if chart_type == "Tendencia por año":
        cat_label = st.selectbox("Categoría para desagregar la tendencia", [
            "Género", "Escolaridad", "Ocupación", "Estado civil", "Municipio"
        ], index=0)
        cat_col = col_options[cat_label]
        top_k = st.slider("Top K categorías (para evitar exceso de líneas)", 3, 30, 10)

        if COL_YEAR not in df_filtered.columns:
            st.warning(f"No se encontró la columna '{COL_YEAR}'.")
        else:
            df_bi = df_filtered[[COL_YEAR, cat_col]].dropna()
            df_bi = filter_unknowns(df_bi, [cat_col], include_unknown)
            df_bi[cat_col] = apply_topn(df_bi[cat_col].astype(str), top_k)
            grp = df_bi.groupby([COL_YEAR, cat_col]).size().reset_index(name="Frecuencia")
            if grp.empty:
                st.info("No hay datos para mostrar.")
            else:
                fig_tr = px.line(
                    grp.sort_values([COL_YEAR, "Frecuencia"]),
                    x=COL_YEAR,
                    y="Frecuencia",
                    color=cat_col,
                    markers=True,
                    title=f"Tendencia por año — desagregado por {cat_label}"
                )
                fig_tr.update_layout(margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig_tr, use_container_width=True)
                with st.expander("Ver tabla"):
                    st.dataframe(grp)
    else:
        c1, c2 = st.columns(2)
        with c1:
            x_label = st.selectbox("Variable en eje X", list(col_options.keys()), index=0)
            top_x = st.slider("Top categorías en X", 3, 50, 12)
        with c2:
            y_label = st.selectbox("Variable en eje Y / Color", list(col_options.keys())[1:], index=0)
            top_y = st.slider("Top categorías en Y", 3, 50, 12)

        x_col = col_options[x_label]
        y_col = col_options[y_label]

        if x_col not in df_filtered.columns or y_col not in df_filtered.columns:
            st.warning("Alguna de las columnas seleccionadas no existe en el DataFrame.")
        else:
            df_bi = df_filtered[[x_col, y_col]].copy()
            df_bi = filter_unknowns(df_bi, [x_col, y_col], include_unknown)
            df_bi[x_col] = df_bi[x_col].astype(str)
            df_bi[y_col] = df_bi[y_col].astype(str)
            df_bi[x_col] = apply_topn(df_bi[x_col], top_x)
            df_bi[y_col] = apply_topn(df_bi[y_col], top_y)

            if chart_type == "Heatmap (X vs Y)":
                ct = pd.crosstab(df_bi[x_col], df_bi[y_col])
                if ct.empty:
                    st.info("No hay datos para mostrar.")
                else:
                    fig_hm = px.imshow(
                        ct.values,
                        x=ct.columns,
                        y=ct.index,
                        text_auto=True,
                        aspect='auto',
                        title=f"Heatmap de frecuencias — {x_label} vs {y_label}"
                    )
                    fig_hm.update_layout(margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(fig_hm, use_container_width=True)
                    with st.expander("Ver tabla (crosstab)"):
                        st.dataframe(ct)

            elif chart_type == "Barras apiladas (conteo)":
                grp = df_bi.groupby([x_col, y_col]).size().reset_index(name="Frecuencia")
                if grp.empty:
                    st.info("No hay datos para mostrar.")
                else:
                    fig_sb = px.bar(
                        grp,
                        x=x_col,
                        y="Frecuencia",
                        color=y_col,
                        barmode='stack',
                        title=f"Barras apiladas (conteo) — {x_label} vs {y_label}"
                    )
                    fig_sb.update_layout(xaxis={'categoryorder': 'total descending'}, margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(fig_sb, use_container_width=True)
                    with st.expander("Ver tabla"):
                        st.dataframe(grp)

            elif chart_type == "Barras apiladas (%)":
                grp = df_bi.groupby([x_col, y_col]).size().reset_index(name="Frecuencia")
                if grp.empty:
                    st.info("No hay datos para mostrar.")
                else:
                    grp["Total_X"] = grp.groupby(x_col)["Frecuencia"].transform('sum')
                    grp["Porcentaje"] = (grp["Frecuencia"] / grp["Total_X"]) * 100
                    fig_sp = px.bar(
                        grp,
                        x=x_col,
                        y="Porcentaje",
                        color=y_col,
                        barmode='stack',
                        title=f"Barras apiladas (%) — {x_label} vs {y_label}",
                        labels={"Porcentaje": "%"}
                    )
                    fig_sp.update_layout(xaxis={'categoryorder': 'total descending'}, margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(fig_sp, use_container_width=True)
                    with st.expander("Ver tabla"):
                        st.dataframe(grp[[x_col, y_col, "Frecuencia", "Porcentaje"]])
                    
with tabs[7]:
    import plotly.graph_objects as go

    st.subheader("Pirámide de edades")

    col_edad   = "Víctima: Edad"
    col_genero = "Víctima: Género"
    df_age = df_filtered.copy()

    df_age[col_genero] = (
        df_age[col_genero]
          .astype(str).str.strip().str.lower()
          .replace({"h": "hombre", "m": "mujer"})          
          .replace({"male": "hombre", "female": "mujer"})
          .map({"hombre": "Hombre", "mujer": "Mujer"})
    )

    df_age[col_edad] = pd.to_numeric(df_age[col_edad], errors="coerce")

    bins   = list(range(0, 90, 5)) + [120]   
    labels = [f"{i}-{i+4}" for i in range(0, 85, 5)] + ["85+"]

    df_age["RangoEdad"] = pd.cut(df_age[col_edad], bins=bins, labels=labels, right=False, include_lowest=True)
    df_age["RangoEdad"] = df_age["RangoEdad"].cat.add_categories(["Sin dato"]).fillna("Sin dato")
    tabla = (
        df_age.groupby(["RangoEdad", col_genero])
              .size()
              .unstack(fill_value=0)[["Hombre", "Mujer"]]  
              .reindex(labels + ["Sin dato"])            
    )

    x_h = -tabla["Hombre"].values
    x_m =  tabla["Mujer"].values
    y   =  tabla.index.tolist()

    max_val  = max(np.abs(x_h).max(), np.abs(x_m).max())
    step     = max(1, int(max_val // 5))  
    tick_pos = np.arange(-max_val, max_val + step, step)
    tick_txt = [str(abs(int(t))) for t in tick_pos] 

    fig = go.Figure()
    fig.add_bar(y=y, x=x_h, name="Hombre", orientation="h")
    fig.add_bar(y=y, x=x_m, name="Mujer",  orientation="h")

    fig.update_layout(
        title="Pirámide de edades",
        barmode="relative",
        xaxis=dict(
            title="Personas",
            tickmode="array",
            tickvals=tick_pos,
            ticktext=tick_txt,
            zeroline=True, zerolinewidth=2
        ),
        yaxis=dict(title="Rangos de edad", categoryorder="array", categoryarray=y),
        bargap=0.1,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_traces(
        hovertemplate="%{y}<br>%{fullData.name}: %{customdata}",
        customdata=[abs(v) for v in x_h]
    )
    fig.data[1].customdata = [abs(v) for v in x_m]

    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Ver tabla"):
        st.dataframe(tabla.reset_index())

with tabs[8]:
    st.subheader("Análisis multivariado")
    st.caption("Explora 2–4 variables categóricas simultáneamente con Parallel Categories, Treemap o Sunburst.")

    col_options = {
        "Año (categoría)": COL_YEAR,
        "Género": COL_GENDER,
        "Escolaridad": COL_EDU,
        "Ocupación": COL_OCC,
        "Estado civil": COL_CIVIL,
        "Municipio": COL_MUNI,
    }

    dims_labels = st.multiselect(
        "Variables (en orden jerárquico/columnas)",
        list(col_options.keys()),
        default=["Género", "Escolaridad", "Ocupación"],
        help="El orden define el path en Treemap/Sunburst y el orden de ejes en Parallel Categories."
    )
    chart_type = st.radio(
        "Tipo de gráfico",
        ["Parallel Categories", "Treemap", "Sunburst"],
        horizontal=True,
        index=0
    )
    top_n_dim = st.slider("Top N por dimensión (agrupar resto en 'Otros')", 3, 30, 10)

    if len(dims_labels) < 2:
        st.info("Selecciona al menos 2 variables para continuar.")
    else:
        dims = [col_options[lbl] for lbl in dims_labels if lbl in col_options]

        df_multi = df_filtered.copy()
        df_multi = filter_unknowns(df_multi, dims, include_unknown)

        for c in dims:
            if c in df_multi.columns:
                df_multi[c] = df_multi[c].astype(str)
                df_multi[c] = apply_topn(df_multi[c], top_n_dim, other_label="Otros")

        df_multi = df_multi.dropna(subset=dims)

        if df_multi.empty:
            st.info("No hay datos para mostrar con los filtros actuales.")
        else:
            if chart_type == "Parallel Categories":
                max_rows = 50_000
                df_pc = df_multi[dims]
                if len(df_pc) > max_rows:
                    df_pc = df_pc.sample(max_rows, random_state=42)

                fig_pc = px.parallel_categories(
                    df_pc,
                    dimensions=dims,
                    title=f"Parallel Categories — {' · '.join(dims_labels)}"
                )
                fig_pc.update_layout(margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig_pc, use_container_width=True)

                with st.expander("Ver combinaciones (Top por frecuencia)"):
                    top_combo = (df_multi[dims]
                                 .value_counts()
                                 .reset_index(name="Frecuencia")
                                 .head(200))
                    st.dataframe(top_combo)

            elif chart_type == "Treemap":
                df_counts = (df_multi[dims]
                             .value_counts()
                             .reset_index(name="Frecuencia"))
                if df_counts.empty:
                    st.info("No hay datos para el Treemap.")
                else:
                    fig_tm = px.treemap(
                        df_counts,
                        path=dims, values="Frecuencia",
                        title=f"Treemap — {' > '.join(dims_labels)}"
                    )
                    fig_tm.update_layout(margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(fig_tm, use_container_width=True)
                    with st.expander("Ver tabla"):
                        st.dataframe(df_counts.sort_values("Frecuencia", ascending=False))

            else:  
                df_counts = (df_multi[dims]
                             .value_counts()
                             .reset_index(name="Frecuencia"))
                if df_counts.empty:
                    st.info("No hay datos para el Sunburst.")
                else:
                    fig_sb = px.sunburst(
                        df_counts,
                        path=dims, values="Frecuencia",
                        title=f"Sunburst — {' > '.join(dims_labels)}"
                    )
                    fig_sb.update_layout(margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(fig_sb, use_container_width=True)
                    with st.expander("Ver tabla"):
                        st.dataframe(df_counts.sort_values("Frecuencia", ascending=False))

st.markdown("---")
st.caption(
    "Notas: " \
    "1) Es recomendable conseguir más datos. " \
    "2) Esta organizado de izquierda a derecha por cantidad de variables. " \
    "3) Opciones para ajustar listas largas a la izquierda. " \
    "4) Ajuste de rango de años por arriba como por abajo al inicio. "
)
