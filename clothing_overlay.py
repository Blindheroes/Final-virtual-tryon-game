"""
Clothing Overlay Module for Virtual Try-On Game
Handles virtual clothing rendering and positioning
"""

import cv2
import numpy as np
import os
import textwrap


class ClothingOverlay:
    def __init__(self):
        """Initialize the clothing overlay system"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.clothing_path = os.path.join(self.base_path, 'clothing')

        # Current clothing items
        self.current_top = None
        self.current_bottom = None
        self.current_right_sleeve = None
        self.current_left_sleeve = None
        self.top_images = []
        self.bottom_images = []
        self.right_sleeve_images = []
        self.left_sleeve_images = []
        self.top_index = 0
        self.bottom_index = 0
        self.current_clothing_images = []
        self.clothing_index = 0

        # Clothing positioning parameters
        self.icon_scale = 0.15
        self.shirt_scale = 0.8
        self.pants_scale = 1.0
        self.sleeve_scale = 1.0

        # Recommendations text
        self.recommendations = {
            'ideal': {
                'male': "Kamu dengan tubuh ideal, hampir semua jenis pakaian dapat digunakan, asalkan tetap memperhatikan "
                        "proporsi tubuh. Pakaian dengan potongan yang pas di badan (well-fitted) sangat dianjurkan. "
                        "Penggunaan celana slim-fit dapat membantu menonjolkan bentuk tubuh yang proporsional.",
                'female': "Kamu dengan tubuh ideal memiliki fleksibilitas lebih dalam memilih model pakaian. Namun, tetap "
                          "disarankan untuk memilih busana yang sesuai dengan aktivitas dan kepribadian. Pakaian yang "
                          "menonjolkan kelebihan tubuh, seperti dress body fit, blouse, atau celana high waist, dapat "
                          "menjadi pilihan utama."
            },
            'over weight': {
                'male': "Pria bertubuh berisi sebaiknya memilih pakaian berwarna gelap seperti hitam, navy, atau abu-abu "
                        "tua untuk memberikan efek ramping. Pakaian dengan potongan yang pas, tidak terlalu longgar atau "
                        "ketat, sangat dianjurkan. Pilihan motif vertikal dapat membantu menciptakan ilusi tubuh yang "
                        "lebih tinggi dan ramping. Celana dengan potongan lurus atau sedikit melebar juga dapat "
                        "memperbaiki proporsi tubuh.",
                'female': "Wanita dengan tubuh berisi disarankan memilih atasan oversize yang tidak terlalu menonjolkan "
                          "lekuk tubuh. Celana high waist dapat memberikan efek pinggang yang lebih ramping. Warna-warna "
                          "netral atau gelap pada pakaian juga dapat membantu menciptakan kesan lebih ramping. Pilihlah "
                          "bahan pakaian yang jatuh dan hindari bahan tebal atau motif besar."
            },
            'under weight': {
                'male': "Pria dengan tubuh kurus dianjurkan memilih pakaian berwarna cerah untuk menciptakan ilusi tubuh "
                        "yang lebih berisi. Penggunaan teknik layering, seperti mengenakan jaket atau blazer di atas kaos, "
                        "juga dapat memberikan volume tambahan pada penampilan. Pilihan pakaian sebaiknya berpotongan slim "
                        "fit, namun tidak terlalu ketat atau longgar. Motif horizontal atau geometris lebih disarankan "
                        "daripada motif garis vertikal, karena dapat mengurangi kesan tubuh yang terlalu ramping. Selain "
                        "itu, bahan pakaian yang agak tebal, seperti denim, juga dapat membantu menambah dimensi pada tubuh.",
                'female': "Wanita dengan tubuh kurus disarankan memilih atasan oversize yang tidak terlalu menonjolkan "
                          "lekuk tubuh. Celana high waist dapat memberikan efek pinggang yang lebih ramping. Warna-warna "
                          "netral atau gelap pada pakaian juga dapat membantu menciptakan kesan lebih ramping. Pilihlah "
                          "bahan pakaian yang jatuh dan hindari bahan tebal atau motif besar."
            }
        }

    def load_clothing_for_body_type(self, gender, body_type, clothing_type='top'):
        """
        Load clothing options for a specific body type and gender

        Args:
            gender: 'male' or 'female'
            body_type: 'ideal', 'under weight', or 'over weight'
            clothing_type: 'top' or 'bottom'

        Returns:
            List of clothing image paths
        """
        # Map internal body type and gender to actual folder names
        body_type_map = {
            'ideal': 'ideal',
            'over weight': 'over',
            'under weight': 'under'
        }

        gender_map = {
            'male': 'man',
            'female': 'woman'
        }

        # Get folder name based on body type and gender
        body_type_folder = body_type_map.get(body_type, 'ideal')
        gender_folder = gender_map.get(gender, 'woman')
        folder_name = f"{body_type_folder}-{gender_folder}"

        clothing_folder = os.path.join(self.clothing_path, folder_name)
        clothing_files = []

        # Check if the path exists
        if os.path.exists(clothing_folder):
            for file in os.listdir(clothing_folder):
                if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    full_path = os.path.join(clothing_folder, file)

                    # Filter files based on clothing type
                    if clothing_type == 'top':
                        # Skip files that are clearly bottoms or sleeves
                        if any(bottom_keyword in file.lower() for bottom_keyword in ['pants', 'jeans', 'skirt', 'bottom', 'highwaist']):
                            continue
                        # Skip arm/sleeve files
                        if any(arm_keyword in file.lower() for arm_keyword in ['arm', 'sleeve']):
                            continue
                    elif clothing_type == 'bottom':
                        # Only include files that are clearly bottoms
                        if any(bottom_keyword in file.lower() for bottom_keyword in ['pants', 'jeans', 'skirt', 'bottom', 'highwaist']):
                            clothing_files.append(full_path)
                    else:
                        clothing_files.append(full_path)

            # If we didn't find any clothing items with the specific filters, include all images as a fallback
            if len(clothing_files) == 0 and clothing_type == 'top':
                for file in os.listdir(clothing_folder):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        # Skip arm/sleeve files
                        if not any(arm_keyword in file.lower() for arm_keyword in ['arm', 'sleeve']):
                            clothing_files.append(
                                os.path.join(clothing_folder, file))

        return clothing_files

    def load_clothing_options(self, gender, body_type):
        """
        Load all available clothing options for a specific body type and gender

        Args:
            gender: 'male' or 'female' 
            body_type: 'ideal', 'under weight', or 'over weight'
        """
        self.top_images = self.load_clothing_for_body_type(
            gender, body_type, 'top')
        self.bottom_images = self.load_clothing_for_body_type(
            gender, body_type, 'bottom')

        # Combine and store all clothing options
        self.current_clothing_images = self.top_images + self.bottom_images
        self.clothing_index = 0

        # Load the first clothing item if available
        if len(self.top_images) > 0:
            self.current_top = self.top_images[0]
        if len(self.bottom_images) > 0:
            self.current_bottom = self.bottom_images[0]

        # Attempt to load sleeve images if they exist
        # This is based on naming conventions - might need adjustment for specific files
        self.current_right_sleeve = self.find_matching_sleeve_file(
            self.current_top, 'right')
        self.current_left_sleeve = self.find_matching_sleeve_file(
            self.current_top, 'left')

    def find_matching_sleeve_file(self, top_file, side):
        """Find matching sleeve file based on top filename"""
        if not top_file:
            return None

        directory = os.path.dirname(top_file)
        base_name = os.path.basename(top_file).split('.')[0]

        # Common naming patterns for sleeve files based on actual files in the project
        patterns = [
            f"{base_name}-arm-{side}.png",  # e.g. pololo-arm-left.png
            f"{base_name}-arm{side}.png",   # e.g. blouse-armleft.png
            f"{base_name}-sleeve-{side}.png",
            f"arm-{side}.png",
            f"arm-{side} 1.png",            # e.g. arm-denim 1.png
            f"arm-{side} 2.png",            # e.g. arm-denim 2.png
            f"dark-shirt-arm-{side}.png",   # For over-woman folder
            f"dark-shirt-arm{side}.png",
            f"dark-denim-arm-{side}.png",   # For over-man folder
            f"dark-denim-arm{side}.png",
            f"{side}-sleeve.png"
        ]

        # First try to find direct matches in the same directory
        for pattern in patterns:
            path = os.path.join(directory, pattern)
            if os.path.exists(path):
                return path

        # Look for any arm file in the directory
        for file in os.listdir(directory):
            if 'arm' in file.lower() and side.lower() in file.lower() and file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                return os.path.join(directory, file)

        return None

    def next_clothing(self, clothing_type='top'):
        """Switch to the next clothing item"""
        if clothing_type == 'top' and len(self.top_images) > 0:
            self.top_index = (self.top_index + 1) % len(self.top_images)
            self.current_top = self.top_images[self.top_index]
            # Update sleeves when top changes
            self.current_right_sleeve = self.find_matching_sleeve_file(
                self.current_top, 'right')
            self.current_left_sleeve = self.find_matching_sleeve_file(
                self.current_top, 'left')
            return self.top_index
        elif clothing_type == 'bottom' and len(self.bottom_images) > 0:
            self.bottom_index = (self.bottom_index +
                                 1) % len(self.bottom_images)
            self.current_bottom = self.bottom_images[self.bottom_index]
            return self.bottom_index
        return -1

    def previous_clothing(self, clothing_type='top'):
        """Switch to the previous clothing item"""
        if clothing_type == 'top' and len(self.top_images) > 0:
            self.top_index = (self.top_index - 1) % len(self.top_images)
            self.current_top = self.top_images[self.top_index]
            # Update sleeves when top changes
            self.current_right_sleeve = self.find_matching_sleeve_file(
                self.current_top, 'right')
            self.current_left_sleeve = self.find_matching_sleeve_file(
                self.current_top, 'left')
            return self.top_index
        elif clothing_type == 'bottom' and len(self.bottom_images) > 0:
            self.bottom_index = (self.bottom_index -
                                 1) % len(self.bottom_images)
            self.current_bottom = self.bottom_images[self.bottom_index]
            return self.bottom_index
        return -1

    def get_current_clothing(self, clothing_type='top'):
        """Get the current clothing item path"""
        if clothing_type == 'top' and len(self.top_images) > 0:
            return self.current_top
        elif clothing_type == 'bottom' and len(self.bottom_images) > 0:
            return self.current_bottom
        elif clothing_type == 'right_sleeve':
            return self.current_right_sleeve
        elif clothing_type == 'left_sleeve':
            return self.current_left_sleeve
        return None

    def roundedRect(self, img, top_left, bottom_right, color, radius):
        """Draw a rounded rectangle on the image"""
        x1, y1 = top_left
        x2, y2 = bottom_right

        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2),
                      color, thickness=cv2.FILLED)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius),
                      color, thickness=cv2.FILLED)

        cv2.circle(img, (x1 + radius, y1 + radius),
                   radius, color, thickness=cv2.FILLED)
        cv2.circle(img, (x2 - radius, y1 + radius),
                   radius, color, thickness=cv2.FILLED)
        cv2.circle(img, (x1 + radius, y2 - radius),
                   radius, color, thickness=cv2.FILLED)
        cv2.circle(img, (x2 - radius, y2 - radius),
                   radius, color, thickness=cv2.FILLED)

    def textWrap(self, frame, text, x, y, max_chars_per_line,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.5,
                 thickness=1,
                 color_text=(255, 255, 255),
                 color_bg=(0, 0, 0),
                 padding=20,
                 alpha=0.5,
                 max_lines=5,
                 radius=20):
        """Draw wrapped text with a rounded rectangle background"""
        wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
        if len(wrapped_lines) > max_lines:
            wrapped_lines = wrapped_lines[:max_lines]

        line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[
            0] for line in wrapped_lines]
        text_width = max(size[0] for size in line_sizes)
        line_height = line_sizes[0][1]

        rect_x1 = x - padding
        rect_y1 = y - line_height - padding
        rect_x2 = x + text_width + padding
        rect_y2 = y + len(wrapped_lines) * line_height + padding

        overlay = frame.copy()

        self.roundedRect(overlay, (rect_x1, rect_y1),
                         (rect_x2, rect_y2), color_bg, radius)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for i, line in enumerate(wrapped_lines):
            line_y = y + i * (line_height + 5)
            cv2.putText(frame, line, (x, line_y), font,
                        font_scale, color_text, thickness)

    def overlay_image(self, background, foreground):
        """
        Overlay foreground image on background using the alpha channel

        Args:
            background: Background image
            foreground: Foreground image with alpha channel

        Returns:
            Combined image
        """
        if foreground.shape[2] == 4:
            # Separate the color and alpha channels
            fg_rgb = foreground[:, :, :3]
            alpha_mask = foreground[:, :, 3] / 255.0

            # Blend the foreground and background
            for c in range(3):
                background[:, :, c] = alpha_mask * fg_rgb[:, :,
                                                          c] + (1 - alpha_mask) * background[:, :, c]
        else:
            background = cv2.addWeighted(background, 1, foreground, 0.5, 0)

        return background

    def sleeves(self, frame, sleeve_img, shoulder, elbow, wrist, scale=1.0, offset=np.array([0, 0])):
        """Render sleeves based on arm positioning"""
        if sleeve_img is None:
            return frame

        h_sleeve, w_sleeve = sleeve_img.shape[:2]
        half_h = h_sleeve // 2

        pts_src_upper = np.float32([
            [0, 0],
            [w_sleeve, 0],
            [w_sleeve / 2, half_h]
        ])
        pts_src_lower = np.float32([
            [0, 0],
            [w_sleeve, 0],
            [w_sleeve / 2, half_h]
        ])

        shoulder = (shoulder + offset).astype(np.float32)
        elbow = (elbow + offset).astype(np.float32)
        wrist = (wrist + offset).astype(np.float32)

        # Create direction vectors
        dir_upper = elbow - shoulder
        dir_upper_norm = np.linalg.norm(dir_upper)
        if dir_upper_norm > 0:
            dir_upper /= dir_upper_norm
            perp_upper = np.array(
                [-dir_upper[1], dir_upper[0]]) * 40  # width = 80

            # Build triangles dynamically
            pts_dst_upper = np.float32([
                shoulder - perp_upper,
                shoulder + perp_upper,
                elbow
            ])

            upper_half = sleeve_img[0:half_h]

            M_upper = cv2.getAffineTransform(pts_src_upper, pts_dst_upper)
            warped_upper = cv2.warpAffine(upper_half, M_upper, (frame.shape[1], frame.shape[0]),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

            frame = self.overlay_image(frame, warped_upper)

        # Lower arm section
        dir_lower = wrist - elbow
        dir_lower_norm = np.linalg.norm(dir_lower)
        if dir_lower_norm > 0:
            dir_lower /= dir_lower_norm
            perp_lower = np.array(
                [-dir_lower[1], dir_lower[0]]) * 40  # width = 80

            pts_dst_lower = np.float32([
                elbow - perp_lower,
                elbow + perp_lower,
                wrist
            ])

            lower_half = sleeve_img[half_h:]

            M_lower = cv2.getAffineTransform(pts_src_lower, pts_dst_lower)
            warped_lower = cv2.warpAffine(lower_half, M_lower, (frame.shape[1], frame.shape[0]),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

            frame = self.overlay_image(frame, warped_lower)

        return frame

    def overlay_clothing(self, frame, body_landmarks=None, gender='female', body_type='ideal'):
        """
        Overlay clothing on the person in the frame

        Args:
            frame: Input frame
            body_landmarks: MediaPipe pose landmarks
            gender: 'male' or 'female'
            body_type: 'ideal', 'under weight', or 'over weight'

        Returns:
            Frame with clothing overlay
        """
        result_frame = frame.copy()

        # Import needed for mediapipe landmarks
        import mediapipe as mp
        mp_pose = mp.solutions.pose

        if body_landmarks and body_landmarks.pose_landmarks:
            h, w = frame.shape[:2]
            lm = body_landmarks.pose_landmarks.landmark

            # Extract landmarks
            left_shoulder = np.array([
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
            ], dtype=np.float32)

            right_shoulder = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
            ], dtype=np.float32)

            left_hip = np.array([
                lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h
            ], dtype=np.float32)

            right_hip = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
            ], dtype=np.float32)

            mid_hip = (left_hip + right_hip) / 2.0

            left_ankle = np.array([
                lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h
            ], dtype=np.float32)

            right_ankle = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
            ], dtype=np.float32)

            mid_ankle = (left_ankle + right_ankle) / 2.0

            right_elbow = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h
            ], dtype=np.float32)

            left_elbow = np.array([
                lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h
            ], dtype=np.float32)

            right_wrist = np.array([
                lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h
            ], dtype=np.float32)

            left_wrist = np.array([
                lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h
            ], dtype=np.float32)

            # Load bottom clothing
            bottom_path = self.get_current_clothing('bottom')
            if bottom_path:
                try:
                    pants = cv2.imread(bottom_path, cv2.IMREAD_UNCHANGED)
                    if pants is not None:
                        # PANTS
                        hip_offset_y = -40
                        pts_dst_pants = np.float32([
                            left_hip + np.array([0, hip_offset_y]),
                            right_hip + np.array([0, hip_offset_y]),
                            mid_ankle
                        ])
                        center_pants = (left_hip + right_hip) / 2.0
                        pts_dst_pants_scaled = center_pants + \
                            self.pants_scale * (pts_dst_pants - center_pants)

                        pants_h, pants_w = pants.shape[:2]
                        pts_src_pants = np.float32([
                            [pants_w * 0.4, 0],
                            [pants_w * 0.6, 0],
                            [pants_w * 0.5, pants_h]
                        ])

                        M_pants = cv2.getAffineTransform(
                            pts_src_pants, pts_dst_pants_scaled)
                        warped_pants = cv2.warpAffine(pants, M_pants, (w, h),
                                                      flags=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_TRANSPARENT)

                        result_frame = self.overlay_image(
                            result_frame, warped_pants)
                except Exception as e:
                    print(f"Error overlaying pants: {e}")

            # Load top clothing
            top_path = self.get_current_clothing('top')
            if top_path:
                try:
                    shirt = cv2.imread(top_path, cv2.IMREAD_UNCHANGED)
                    if shirt is not None:
                        # SHIRT
                        offset_shirt = -40
                        pts_dst_shirt = np.float32([
                            left_shoulder + np.array([0, offset_shirt]),
                            right_shoulder + np.array([0, offset_shirt]),
                            right_hip,
                            left_hip
                        ])
                        center_shirt = (left_shoulder + right_shoulder) / 2.0
                        pts_dst_shirt_scaled = center_shirt + \
                            self.shirt_scale * (pts_dst_shirt - center_shirt)

                        shirt_h, shirt_w = shirt.shape[:2]
                        pts_src_shirt = np.float32([
                            [shirt_w * 0.2, 0],          # kiri atas
                            [shirt_w * 0.8, 0],          # kanan atas
                            [shirt_w * 0.7, shirt_h],    # kanan bawah
                            [shirt_w * 0.3, shirt_h]     # kiri bawah
                        ])

                        M_shirt = cv2.getPerspectiveTransform(
                            pts_src_shirt, pts_dst_shirt_scaled)
                        warped_shirt = cv2.warpPerspective(shirt, M_shirt, (w, h),
                                                           flags=cv2.INTER_LINEAR,
                                                           borderMode=cv2.BORDER_TRANSPARENT)

                        # Draw right sleeve
                        right_sleeve_path = self.get_current_clothing(
                            'right_sleeve')
                        if right_sleeve_path:
                            try:
                                right_sleeve = cv2.imread(
                                    right_sleeve_path, cv2.IMREAD_UNCHANGED)
                                if right_sleeve is not None:
                                    result_frame = self.sleeves(
                                        result_frame, right_sleeve,
                                        right_shoulder, right_elbow, right_wrist,
                                        scale=1.0, offset=np.array([0, -10])
                                    )
                            except Exception as e:
                                print(f"Error overlaying right sleeve: {e}")

                        # Draw left sleeve
                        left_sleeve_path = self.get_current_clothing(
                            'left_sleeve')
                        if left_sleeve_path:
                            try:
                                left_sleeve = cv2.imread(
                                    left_sleeve_path, cv2.IMREAD_UNCHANGED)
                                if left_sleeve is not None:
                                    result_frame = self.sleeves(
                                        result_frame, left_sleeve,
                                        left_shoulder, left_elbow, left_wrist,
                                        scale=1.0, offset=np.array([0, -10])
                                    )
                            except Exception as e:
                                print(f"Error overlaying left sleeve: {e}")

                        # Draw shirt body (after sleeves)
                        result_frame = self.overlay_image(
                            result_frame, warped_shirt)
                except Exception as e:
                    print(f"Error overlaying shirt: {e}")

            # Add fashion recommendations
            if body_type in self.recommendations and gender in self.recommendations[body_type]:
                recommendation_text = self.recommendations[body_type][gender]
                x, y = 10, 400
                max_chars_per_line = 75
                self.textWrap(result_frame, recommendation_text,
                              x, y, max_chars_per_line, font_scale=0.5)

        return result_frame
