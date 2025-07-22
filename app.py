# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import math
import os

# --- Configuration ---
APP_TITLE = "NrityaNirman: AI-Powered Precision for Indian Dance Technique"
APP_SUBTITLE = "Analyze your Bharatanatyam Tatta Adavu with AI feedback."

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Reference Pose for Tatta Adavu (Simplified for MVP) ---
# These are approximate ideal angles/positions for a key moment in Tatta Adavu (e.g., Aramandi)
# In a real system, these would come from expert-annotated data or a more complex model.
# Using specific joint indices from MediaPipe:
# LEFT_HIP = 23, LEFT_KNEE = 25, LEFT_ANKLE = 27
# RIGHT_HIP = 24, RIGHT_KNEE = 26, RIGHT_ANKLE = 28
# LEFT_SHOULDER = 11, LEFT_ELBOW = 13, LEFT_WRIST = 15
# RIGHT_SHOULDER = 12, RIGHT_ELBOW = 14, RIGHT_WRIST = 16

# Ideal Aramandi (half-sit) posture angles (approximate, in degrees)
# Angles are calculated between three points (e.g., hip-knee-ankle)
IDEAL_LEFT_KNEE_ANGLE = 100  # Slightly bent, forming approx 100-110 degrees
IDEAL_RIGHT_KNEE_ANGLE = 100
IDEAL_HIP_ANGLE = 170      # Hip-shoulder-knee, aiming for straight torso relative to thigh
IDEAL_ARM_ANGLE_BENT = 90  # For bent arm position (e.g., in Pataka hasta)
IDEAL_ARM_ANGLE_STRAIGHT = 180 # For straight arm position

# --- Helper Functions ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points (a, b, c) with b as the vertex."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_feedback(user_angles, ideal_angles, threshold=10):
    """
    Compares user angles to ideal angles and provides simple feedback.
    Returns a tuple: (score, feedback_messages)
    """
    total_score = 0
    feedback_messages = []

    # Knee angles
    left_knee_diff = abs(user_angles['left_knee'] - ideal_angles['left_knee'])
    right_knee_diff = abs(user_angles['right_knee'] - ideal_angles['right_knee'])

    if left_knee_diff > threshold:
        feedback_messages.append(f"Adjust left knee: Current {user_angles['left_knee']:.0f}°, Ideal {ideal_angles['left_knee']}°.")
    else:
        total_score += 1
    if right_knee_diff > threshold:
        feedback_messages.append(f"Adjust right knee: Current {user_angles['right_knee']:.0f}°, Ideal {ideal_angles['right_knee']}°.")
    else:
        total_score += 1

    # Hip angles (for torso alignment)
    left_hip_diff = abs(user_angles['left_hip'] - ideal_angles['left_hip'])
    right_hip_diff = abs(user_angles['right_hip'] - ideal_angles['right_hip'])

    if left_hip_diff > threshold:
        feedback_messages.append(f"Adjust left hip/torso: Current {user_angles['left_hip']:.0f}°, Ideal {ideal_angles['left_hip']}°.")
    else:
        total_score += 1
    if right_hip_diff > threshold:
        feedback_messages.append(f"Adjust right hip/torso: Current {user_angles['right_hip']:.0f}°, Ideal {ideal_angles['right_hip']}°.")
    else:
        total_score += 1

    # Simple arm angle check (assuming a bent arm for Pataka hasta)
    left_elbow_diff = abs(user_angles['left_elbow'] - ideal_angles['left_elbow'])
    right_elbow_diff = abs(user_angles['right_elbow'] - ideal_angles['right_elbow'])

    if left_elbow_diff > threshold:
        feedback_messages.append(f"Adjust left arm: Current {user_angles['left_elbow']:.0f}°, Ideal {ideal_angles['left_elbow']}°.")
    else:
        total_score += 1
    if right_elbow_diff > threshold:
        feedback_messages.append(f"Adjust right arm: Current {user_angles['right_elbow']:.0f}°, Ideal {ideal_angles['right_elbow']}°.")
    else:
        total_score += 1
        
    # Overall score based on how many checks passed
    overall_score = (total_score / 6) * 100 # 6 checks in total

    if not feedback_messages:
        feedback_messages.append("Excellent! Your pose looks good.")
    elif overall_score >= 70:
        feedback_messages.insert(0, "Good attempt! Here are some areas for refinement:")
    else:
        feedback_messages.insert(0, "Keep practicing! Focus on these areas:")

    return overall_score, feedback_messages

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title=APP_TITLE)

st.title(APP_TITLE)
st.markdown(APP_SUBTITLE)

st.sidebar.header("Upload Your Dance Video")
uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

st.sidebar.header("Analysis Settings")
selected_dance_form = st.sidebar.selectbox("Select Dance Form", ["Bharatanatyam", "Bollywood (Basic)"])
if selected_dance_form == "Bharatanatyam":
    selected_adavu = st.sidebar.selectbox("Select Adavu", ["Tatta Adavu (Basic Aramandi)"])
else:
    selected_adavu = st.sidebar.selectbox("Select Movement", ["Basic Energy & Timing"])

# Placeholder for reference video (if we had one)
# st.sidebar.video("path/to/reference_tatta_adavu.mp4") # Example

st.markdown("---")

if uploaded_file is not None:
    st.subheader("Your Uploaded Video")
    
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Display the uploaded video
    st.video(video_path)

    st.subheader("AI Analysis Results")
    
    # Create placeholders for processed video and feedback
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Processed Video with Pose Estimation")
        video_placeholder = st.empty()
    with col2:
        st.markdown("#### Technique Feedback")
        feedback_placeholder = st.empty()
        score_placeholder = st.empty()
        
    st.markdown("---")
    st.info("Analysis in progress... This may take a moment depending on video length.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use a temporary file for the output video
    output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    feedback_per_frame = []
    pose_scores = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates for key joints
                # Left side
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Right side
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angles
                user_angles = {
                    'left_knee': calculate_angle(left_hip, left_knee, left_ankle),
                    'right_knee': calculate_angle(right_hip, right_knee, right_ankle),
                    'left_hip': calculate_angle(left_shoulder, left_hip, left_knee),
                    'right_hip': calculate_angle(right_shoulder, right_hip, right_knee),
                    'left_elbow': calculate_angle(left_shoulder, left_elbow, left_wrist),
                    'right_elbow': calculate_angle(right_shoulder, right_elbow, right_wrist)
                }
                
                # Define ideal angles for Tatta Adavu's Aramandi
                # These are simplified and would be more nuanced in a full system
                ideal_tatta_adavu_angles = {
                    'left_knee': IDEAL_LEFT_KNEE_ANGLE,
                    'right_knee': IDEAL_RIGHT_KNEE_ANGLE,
                    'left_hip': IDEAL_HIP_ANGLE,
                    'right_hip': IDEAL_HIP_ANGLE,
                    'left_elbow': IDEAL_ARM_ANGLE_BENT,
                    'right_elbow': IDEAL_ARM_ANGLE_BENT
                }

                score, feedback_msgs = get_feedback(user_angles, ideal_tatta_adavu_angles)
                feedback_per_frame.append(feedback_msgs)
                pose_scores.append(score)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing_styles.get_default_pose_landmarks_style())

            except Exception as e:
                # Handle cases where no pose is detected or other errors
                feedback_per_frame.append(["Could not detect pose. Ensure full body is visible."])
                pose_scores.append(0)
                # print(f"Error processing frame: {e}") # For debugging

            out.write(image) # Write frame to output video
            frame_count += 1

            # Update placeholders (can optimize to update less frequently for performance)
            if frame_count % 5 == 0: # Update every 5 frames
                video_placeholder.image(image, channels="BGR", use_column_width=True)
                feedback_placeholder.markdown("\n".join([f"- {msg}" for msg in feedback_msgs]))
                score_placeholder.metric("Current Pose Score", f"{score:.1f}%")

    cap.release()
    out.release()
    
    # Calculate average score
    if pose_scores:
        avg_score = np.mean(pose_scores)
        st.success(f"Analysis Complete! Your average technique score for this performance is: **{avg_score:.1f}%**")
    else:
        avg_score = 0
        st.warning("No poses detected throughout the video. Please ensure the dancer is fully visible.")

    # Display the processed video at the end
    st.subheader("Full Processed Video")
    st.video(output_video_path)

    # Clean up temporary files
    os.unlink(video_path)
    os.unlink(output_video_path)

else:
    st.info("Please upload a video to start the dance technique analysis.")

st.markdown("---")
st.markdown("""
    **Note on Analysis:** This is a simplified demonstration focusing on basic pose estimation and angle comparison.
    A full-fledged system would involve:
    * More sophisticated AI models trained on diverse dance data.
    * Detailed analysis of *mudras*, footwork, rhythm, and fluidity.
    * Comparison against a comprehensive library of expert-annotated reference movements.
    * Advanced feedback mechanisms and progress tracking stored in a database.
""")

