"""Run with: python -m streamlit run app.py"""
import os
from datetime import datetime, timezone

import streamlit as st
from groq import Groq
from streamlit.errors import StreamlitSecretNotFoundError

from tutor import (
    DIRECTION_LABELS, HISTORY_LIMIT, MAX_INPUT, TOPICS,
    Attempt, Practice, TutorError, TutorService, supported_models,
)


@st.cache_resource(max_entries=8)
def get_client(api_key: str) -> Groq:
    # Include api_key in the cache key; never use an underscore-prefixed argument here.
    try:
        return Groq(api_key=api_key, timeout=30.0, max_retries=1)
    except ImportError as error:
        raise TutorError(
            'Proxy üçün əlavə paket tələb olunur. İcra edin: python -m pip install "httpx[socks]"'
        ) from error


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def get_models(api_key: str) -> list[str]:
    return supported_models(get_client(api_key))


def configured_key() -> str:
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def feedback(attempt: Attempt) -> None:
    data = attempt.evaluation
    with st.container(border=True):
        st.caption("Son yoxlama · " + DIRECTION_LABELS[attempt.direction])
        st.text("Cümlə: " + attempt.sentence)
        st.text("Sizin cavabınız: " + attempt.answer)
        renderer = {"Düzgün": st.success, "Qismən doğrudur": st.warning, "Səhv": st.error}
        renderer[data["result"]](data["result"])
        st.markdown("**Düzgün variant nümunəsi**")
        st.text(data["ideal_translation"])
        st.markdown("**İzah**")
        st.text(data["explanation"])
        for correction in data["corrections"]:
            st.text("• " + correction)
        if data["naturalness"]:
            st.markdown("**Daha təbii ifadə haqqında qeyd**")
            st.text(data["naturalness"])


def history_view(practice: Practice) -> None:
    if not practice.history:
        return
    with st.expander(f"Nəticə tarixçəsi ({len(practice.history)})"):
        st.caption(f"Bu sessiyanın son {HISTORY_LIMIT} cavabı saxlanılır; aşağıda son 20 cavab göstərilir.")
        st.dataframe(
            [{"İstiqamət": DIRECTION_LABELS[a.direction], "Cümlə": a.sentence,
              "Cavabınız": a.answer, "Nəticə": a.evaluation["result"],
              "Düzgün nümunə": a.evaluation["ideal_translation"]}
             for a in reversed(practice.history[-20:])],
            hide_index=True,
        )
        st.download_button(
            "Tarixçəni endir (JSON)", data=practice.export(),
            file_name="a1-tarixce-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json",
            mime="application/json", key="download_history",
        )
        st.caption("Səhifə yenilənəndə və ya sessiya bitəndə nəticələr itə bilər. Saxlamaq üçün tarixçəni endirin.")


def main() -> None:
    st.set_page_config(page_title="A1 İngilis–Rus məşqi", page_icon="🔤", layout="centered")
    st.title("🔤 A1 İngilis–Rus məşqi")
    st.caption("Qısa cümləni tərcümə edin, cavabınızı yoxlayın və izahı oxuyun.")

    if "practice" not in st.session_state:
        st.session_state.practice = Practice()
    practice = st.session_state.practice

    with st.sidebar:
        st.subheader("Məşq ayarları")
        direction = st.radio("Tərcümə istiqaməti", list(DIRECTION_LABELS),
                             format_func=DIRECTION_LABELS.get, key="direction")
        topic_label = st.selectbox("Mövzu", list(TOPICS), key="topic")
        auto_next = st.toggle("Yoxlamadan sonra yeni cümlə", value=True, key="auto_next",
                              help="Söndürsəniz, nəticəni oxuyub növbəti cümləyə özünüz keçirsiniz.")
        with st.expander("Qiymətləndirmə qaydası"):
            st.write("Düzgün: məna və qrammatika uyğundur. Qismən doğrudur: məna əsasən qalır, düzəliş lazımdır. "
                     "Səhv: əsas məna dəyişib və ya cavab uyğun dildə deyil.")
            st.write("Sinonimlər və düzgün söz sırası variantları qəbul edilir. AI bəzən səhv qiymətləndirə bilər.")
        api_key = configured_key()
        if not api_key:
            api_key = st.text_input("Groq API açarı", type="password", key="api_key_input",
                                    help="Açarı burada daxil edə və ya .streamlit/secrets.toml faylında saxlaya bilərsiniz.").strip()

    if not api_key:
        st.info("Başlamaq üçün yan paneldə Groq API açarını daxil edin.")
        st.code('GROQ_API_KEY = "gsk_..."', language="toml")
        st.caption("Bu sətri alternativ olaraq .streamlit/secrets.toml faylına yaza bilərsiniz.")
        st.stop()

    with st.sidebar.expander("Model ayarları"):
        if st.button("Model siyahısını yenilə", key="refresh_models"):
            get_models.clear(api_key)
        try:
            models = get_models(api_key)
        except TutorError as error:
            st.error(str(error))
            st.stop()
        # A deleted model must not leave a stale widget selection behind.
        if st.session_state.get("model") not in models:
            st.session_state["model"] = models[0]
        model = st.selectbox("AI modeli", models, key="model")
        st.caption("Siyahıda bu hesabda görünən, proqramın dəstəklədiyi mətn modelləri göstərilir.")

    service = TutorService(get_client(api_key), model)
    practice.set_context(direction, TOPICS[topic_label])
    # API calls happen only at the first exercise or on an explicit user action.
    if practice.current is None and not practice.initial_attempted:
        with st.spinner("A1 cümləsi hazırlanır..."):
            practice.next_exercise(service)

    if st.button("🎯 Yeni cümlə", key="next_sentence",
                 help="Cavablandırılmamış cümləni keçmək statistikaya təsir etmir."):
        with st.spinner("Yeni cümlə hazırlanır..."):
            practice.next_exercise(service)

    total = sum(practice.counts.values())
    columns = st.columns(4)
    columns[0].metric("Yoxlanılan", total)
    columns[1].metric("Düzgün", practice.counts["Düzgün"])
    columns[2].metric("Düzgün cavab faizi", f"{100 * practice.counts['Düzgün'] / total:.0f}%" if total else "—")
    columns[3].metric("Ardıcıl düzgün", practice.streak)
    st.caption(f"Qismən doğru: {practice.counts['Qismən doğrudur']} · Səhv: {practice.counts['Səhv']}")

    error_slot = st.empty()
    if practice.error:
        error_slot.error(practice.error)
    if practice.last:
        feedback(practice.last)

    exercise = practice.current
    if exercise:
        if exercise.answered:
            st.info("Bu cümlə yoxlanılıb. Davam etmək üçün “Yeni cümlə” düyməsini sıxın.")
        else:
            st.markdown("**Tərcümə edin**")
            st.text(exercise.sentence)
            target = "rus dilinə" if exercise.direction == "en_ru" else "ingilis dilinə"
            st.caption(f"Cümləni {target} tərcümə edin. Enter düyməsi cavabı göndərir.")
            # Keep draft on failed evaluation. A new UUID creates an empty field only
            # after a new exercise has been successfully generated.
            with st.form("translation_" + exercise.id, clear_on_submit=False):
                answer = st.text_input("Tərcüməniz", key="answer_" + exercise.id, max_chars=MAX_INPUT)
                submitted = st.form_submit_button("✅ Cavabı yoxla", type="primary")
            if submitted:
                if not answer.strip():
                    st.warning("Əvvəlcə tərcümənizi yazın.")
                else:
                    with st.spinner("Tərcümə yoxlanılır..."):
                        if practice.submit(service, answer, auto_next):
                            st.rerun()
                    if practice.error:
                        error_slot.error(practice.error)

    history_view(practice)


if __name__ == "__main__":
    main()
