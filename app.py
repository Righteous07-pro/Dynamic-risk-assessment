import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Motor Insurance Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboards import main as run_dashboard


def main():
    """Main entry point for the Streamlit application."""
    try:
        run_dashboard()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback

        st.error(f"Error details: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
