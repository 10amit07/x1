import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import pandas as pd
from PIL import Image
import io
import json
import re
import time
from datetime import datetime
from google.oauth2 import service_account
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Initialize Streamlit page configuration
st.set_page_config(page_title="Fresh Produce Analyzer", layout="wide")


hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {display: none !important;}
        header {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none !important;}
        .css-1lsmgbg.egzxvld1 {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize session states
if 'produce_data' not in st.session_state:
    st.session_state.produce_data = []

try:
    # Initialize Google Cloud credentials
    credentials_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    project_id = st.secrets["GOOGLE_CLOUD_PROJECT"]
    
    # Initialize Vertex AI
    vertexai.init(project=project_id, location="us-central1", credentials=credentials)
    
    # Initialize the Gemini model
    model = GenerativeModel("gemini-1.5-flash-002")
    st.success("Model loaded successfully")

except Exception as e:
    st.error(f"Error loading Google Cloud credentials: {str(e)}")
    st.stop()

def create_prompt_template():
    return """Analyze this image of produce (fruits/vegetables) and provide:

1. Produce Identification:
   Identify the specific fruit or vegetable shown.

2. Freshness Assessment (1-10 scale):
   Evaluate visual indicators including color, texture, blemishes. Vegetables tend to stay fresh for longer periods of time, so a higher score is expected.

3. Expected Shelf Life:
   Predict remaining days of freshness.
   
4. Confidence Score:
   Provide reliability percentage of assessment.

5. Visual Indicators Observed:
   List key visual cues used in assessment.

Present results in this format:
Produce: [name]
Freshness Score: [1-10]
Expected Lifespan (Days): [number]
Confidence Score: [percentage]
Key Indicators: [bullet points]"""

def analyze_image(image):
    try:
        start_time = time.time()
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        prompt = create_prompt_template()
        
        # Create proper Part object
        image_part = Part.from_data(img_byte_arr, mime_type="image/png")
        
        # Send both prompt and image as parts
        response = model.generate_content(
            [prompt, image_part],
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0.1,
                "top_p": 1,
                "top_k": 32
            }
        )
        
        # detection_time = (time.time() - start_time) * 1000
        # if detection_time > 1000:
        #     st.warning(f"Detection time exceeded threshold: {detection_time:.2f}ms")
        # else:
        #     st.success(f"Detection time: {detection_time:.2f}ms")
        
        analysis = parse_analysis_response(response.text)
        current_time = datetime.now().astimezone().isoformat()
        analysis['timestamp'] = current_time
        
        return analysis
        
    except Exception as e:
        st.error("Error analyzing image")
        return None

def parse_analysis_response(response_text):
    details = {
        "Sl No": len(st.session_state.produce_data) + 1,
        "Timestamp": datetime.now().astimezone().isoformat(),
        "Produce": "Not specified",
        "Freshness": 0,
        "Expected Lifespan (Days)": 0,
        "Visual Indicators": []
    }
    
    current_section = None
    
    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Produce:'):
            details["Produce"] = line.split(':', 1)[1].strip()
        elif line.startswith('Freshness Score:'):
            score = line.split(':', 1)[1].strip()
            details["Freshness"] = int(score) if score.isdigit() else 0
        elif line.startswith('Expected Lifespan'):
            lifespan = line.split(':', 1)[1].strip()
            match = re.search(r'(\d+)', lifespan)
            details["Expected Lifespan (Days)"] = int(match.group(1)) if match else 0
        elif line.startswith('Key Indicators:'):
            current_section = "Indicators"
        elif current_section == "Indicators" and line.startswith('-'):
            details["Visual Indicators"].append(line.lstrip('- '))
    
    return details

def generate_pdf_report(data, filename):
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("Fresh Produce Analysis Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 20))  # Add more space after title
        
        # Report generation date
        date_text = Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(date_text)
        elements.append(Spacer(1, 12))
        
        # Convert data to table format
        table_data = [["Sl No", "Timestamp", "Produce", "Freshness", "Expected Lifespan (Days)"]]
        for item in data:
            row = [
                str(item["Sl No"]),
                item["Timestamp"].split('T')[0],  # Format timestamp to show only date
                item["Produce"],
                str(item["Freshness"]),
                str(item["Expected Lifespan (Days)"])
            ]
            table_data.append(row)
        
        # Create and style table
        table = Table(table_data, repeatRows=1)  # Repeat header row on each page
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
        ]))
        
        elements.append(table)
        doc.build(elements)
        return True
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return False

def update_produce_data(analysis):
    table_row = {
        "Sl No": len(st.session_state.produce_data) + 1,
        "Timestamp": analysis["timestamp"],
        "Produce": analysis["Produce"],
        "Freshness": analysis["Freshness"],
        "Expected Lifespan (Days)": analysis["Expected Lifespan (Days)"]
    }
    st.session_state.produce_data.append(table_row)

def main():
    st.title("Fresh Produce Analyzer")
    
    # Image input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_type = st.radio("Choose input method:", ["Upload Image", "Camera Input"])
        
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image of produce", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            img_file = st.camera_input("Take a picture of produce")
            if img_file is not None:
                image = Image.open(img_file)
                
        if 'image' in locals():
            # Resize image for display
            max_width = 300
            ratio = max_width / image.width
            new_size = (max_width, int(image.height * ratio))
            resized_image = image.resize(new_size)
            st.image(resized_image, caption="Input Image")
            
            if st.button("Analyze Produce"):
                with st.spinner("Analyzing image..."):
                    analysis = analyze_image(image)
                    if analysis:
                        update_produce_data(analysis)
                        
                        st.subheader("Analysis Results:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"*Produce:* {analysis['Produce']}")
                            st.write(f"*Freshness Score:* {analysis['Freshness']}/10")
                        with col2:
                            st.write(f"*Expected Shelf Life:* {analysis['Expected Lifespan (Days)']} days")
                            if analysis['Visual Indicators']:
                                st.write("*Visual Indicators:*")
                                for indicator in analysis['Visual Indicators']:
                                    st.write(f"- {indicator}")
    
    # Display produce history
    st.subheader("Produce Analysis History")
    if st.session_state.produce_data:
        df = pd.DataFrame(st.session_state.produce_data)
        st.dataframe(df, use_container_width=True)
        
        # Report generation section
        st.subheader("Generate Report")
        if st.button("Generate PDF Report"):
            pdf_filename = f"produce_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            if generate_pdf_report(st.session_state.produce_data, pdf_filename):
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
                st.success("PDF report generated successfully!")
    else:
        st.info("No produce analyzed yet.")

if _name_ == "_main_":
    main()
