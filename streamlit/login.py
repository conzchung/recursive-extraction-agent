import streamlit as st
import bcrypt
import time
import json
from pathlib import Path
from streamlit_cookies_controller import CookieController

from create_account import verify_user

controller = CookieController(key="app_cookies")

# =========================
# Styling
# =========================
hide_streamlit_style = """
<style>
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 0.5rem !important;
    margin-top: 0rem !important;
    margin-bottom: 2rem !important;
}
div[data-testid="stDecoration"] {
    visibility: hidden;
    height: 0%;
    position: fixed;
}
div[data-testid="stStatusWidget"] {
    visibility: hidden;
    height: 0%;
    position: fixed;
}
#root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# =========================
# Helper functions
# =========================
def check_cookie_auth():
    """Check if the user has a valid authentication cookie and restore session state.

    Reads 'auth_token', 'username', and 'user_role' cookies. If all are present
    and the auth token is valid, populates st.session_state so the user bypasses
    the login form.

    Returns:
        bool: True if the user was authenticated via cookie, False otherwise.
    """
    time.sleep(0.3)
    auth_cookie = controller.get('auth_token')
    if auth_cookie and auth_cookie == 'authenticated':
        username = controller.get('username')
        role = controller.get('user_role')
        if username and role:
            st.session_state['login_status'] = True
            st.session_state['username'] = username
            st.session_state['user_role'] = role
            return True
    return False


def check_password(username, password):
    """Verify a username/password pair against Cosmos DB.

    Delegates to the create_account.verify_user helper. If the authentication
    service is unreachable, an error is displayed and the login attempt fails.

    Args:
        username: The username entered by the user (case-insensitive).
        password: The plaintext password to verify.

    Returns:
        tuple: (is_authenticated, role) where is_authenticated is a bool and
            role is the user's role string or None on failure.
    """
    username = username.lower()

    try:
        # Try to verify against Cosmos DB
        is_authenticated, role = verify_user(username, password)
        if is_authenticated:
            return True, role
    except Exception as e:
        st.error(f"Authentication service unavailable: {e}")
        return False, None

    return False, None


# =========================
# Login page
# =========================
def login():
    """Render the login page and handle form submission.

    First checks for an existing auth cookie. If none is found, displays
    a login form. On successful authentication, sets session state and
    cookies (30-day expiry) then triggers a page rerun.
    """
    if check_cookie_auth():
        st.rerun()
        
    st.set_page_config(
        page_title="AI Extraction & Validation",
        page_icon="🐱",
        layout="centered",
    )
        
    st.markdown("<h2 style='text-align: center;'>Welcome to AI Extraction & Validation 🐱✨</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Please enter your username and password to login. 🔑</h5>", unsafe_allow_html=True)
    
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login", width="stretch")
        
        if login_btn:
            username = username.lower()
            is_authenticated, role = check_password(username, password)
            if is_authenticated:
                st.session_state['login_status'] = True
                st.session_state['username'] = username
                st.session_state['user_role'] = role
                
                try:
                    controller.set('auth_token', 'authenticated', max_age=30*24*60*60)
                    controller.set('username', username, max_age=30*24*60*60)
                    controller.set('user_role', role, max_age=30*24*60*60)
                except Exception as e:
                    st.error(f"Error setting cookies: {str(e)}")
                
                time.sleep(0.5)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

# =========================
# Logout page
# =========================
def logout():
    """Render the logout confirmation page and handle sign-out.

    Displays a confirmation button. When clicked, removes all auth cookies,
    clears session state, and redirects the user back to the login page.
    """
    st.set_page_config(
        page_title="AI Extraction & Validation",
        page_icon="🐱",
        layout="centered",
    )
    
    st.markdown("<h2 style='text-align: center;'>🚪 Logout</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Are you sure you want to log out?</h5>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        logging_out = st.button("Yes, Log Out", width='stretch')
        
    if logging_out:
        st.success("You have been logged out successfully!")
        cookies_to_remove = ['auth_token', 'username', 'user_role']
        for cookie in cookies_to_remove:
            if controller.get(cookie) is not None:
                controller.remove(cookie)
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.session_state['login_status'] = False
        time.sleep(1)
        st.rerun()


# =========================
# Pages
# =========================
home_page = st.Page("homepage.py", title="Home", icon="🏠")

extract_partition = st.Page("pages/extract_partition.py", title="Documents Partition", icon="✂️")
extract_config = st.Page("pages/extract_config.py", title="Configuration Setup", icon="⚙️")
extract_docs = st.Page("pages/extract_docs.py", title="Document Extraction", icon="📤")
extract_monitor = st.Page("pages/extract_monitor.py", title="Monitor Progress", icon="📊")
extract_results = st.Page("pages/extract_results.py", title="Results Query", icon="🔍")

validate_rules = st.Page("pages/validate_rules.py", title="Validation Rules", icon="📏")
validate_docs = st.Page("pages/validate_docs.py", title="Validate Documents", icon="🛡️")
validate_monitor = st.Page("pages/validate_monitor.py", title="Validation Progress", icon="⏳")
validate_results = st.Page("pages/validate_results.py", title="Validation Results", icon="📋")

login_page = st.Page(login, title="Login", icon="🔑")
logout_page = st.Page(logout, title="Logout", icon="🚪")


if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False

if st.session_state['login_status']:
    user_role = st.session_state['user_role']
    if user_role == 'admin':
        pages = st.navigation(
            {
                "🏠 Home": [home_page],
                "📤 Extraction": [
                    extract_config,
                    extract_docs,
                    extract_monitor,
                    extract_results,
                    extract_partition,
                ],
                "🛡️ Validation": [
                    validate_rules,
                    validate_docs,
                    validate_monitor,
                    validate_results,
                ],
                "👤 Account": [logout_page],
            },
            position="top",
        )
else:
    pages = st.navigation([login_page])

pages.run()