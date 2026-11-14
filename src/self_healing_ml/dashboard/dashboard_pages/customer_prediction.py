
import json
import urllib.error
import urllib.request

import streamlit as st

from self_healing_ml.dashboard.config import API_URL
from self_healing_ml.dashboard.components import section_heading


def get_prediction(customer_data):
    """Send customer details to the deployed prediction API."""

    payload = json.dumps({"customer": customer_data}).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def render(data=None):
    """Render the customer churn prediction page."""

    section_heading(
        "Customer Churn Prediction",
        "Enter customer information to predict whether the customer is likely to leave the service.",
    )

    st.info(
        "Predictions are generated using the deployed Self-Healing ML API."
    )

    # CUSTOMER DETAILS FORM

    with st.form("customer_prediction_form"):

        st.markdown("### Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            partner = st.selectbox(
                "Has Partner?",
                ["Yes", "No"],
            )

        with col2:
            dependents = st.selectbox(
                "Has Dependents?",
                ["Yes", "No"],
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12,
                step=1,
            )

        st.markdown("### Services Information")

        col1, col2 = st.columns(2)

        with col1:
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"],
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )

        st.markdown("### Contract and Billing Information")

        col1, col2 = st.columns(2)

        with col1:
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col2:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=1.0,
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True,
        )

    # API PREDICTION

    if submitted:

        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner(
            "Connecting to the prediction API. "
            "This may take a little longer if the server is waking up..."
        ):
            try:
                result = get_prediction(customer_data)

                st.markdown("---")
                st.markdown("### Prediction Result")

                prediction = result.get(
                    "prediction",
                    result.get("predicted_churn"),
                )

                probability = result.get(
                    "probability",
                    result.get("churn_probability"),
                )

                if prediction is not None:
                    prediction_text = str(prediction).strip().lower()

                    if prediction_text in ["yes", "1", "true", "churn"]:
                        st.error("Customer is likely to churn.")

                    elif prediction_text in ["no", "0", "false", "stay"]:
                        st.success("Customer is unlikely to churn.")

                    else:
                        st.info(f"Prediction: {prediction}")

                if probability is not None:
                    try:
                        probability = float(probability)

                        # Handle probabilities returned as percentages.
                        if probability > 1:
                            probability /= 100

                        st.metric(
                            "Churn Probability",
                            f"{probability:.2%}",
                        )

                        st.progress(
                            max(0.0, min(1.0, probability))
                        )

                    except (TypeError, ValueError):
                        st.write(
                            f"Churn Probability: {probability}"
                        )

                # Keep the technical response available for debugging.
                with st.expander("View Technical API Response"):
                    st.json(result)

            except urllib.error.HTTPError as error:
                error_details = error.read().decode(
                    "utf-8",
                    errors="replace",
                )

                st.error(
                    f"API returned an HTTP error ({error.code})."
                )
                st.code(error_details)

            except urllib.error.URLError as error:
                st.error(
                    "Could not connect to the prediction API. "
                    "Please check the API deployment and try again."
                )
                st.caption(str(error.reason))

            except TimeoutError:
                st.error(
                    "The prediction API took too long to respond. "
                    "Please try again."
                )

            except (ValueError, json.JSONDecodeError):
                st.error("The API returned an invalid response.")

            except Exception as error:
                st.error(
                    f"An unexpected error occurred: {error}"
                )
