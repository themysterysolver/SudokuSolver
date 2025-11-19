import os
import cv2
import torch
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import imutils
import matplotlib.pyplot as plt

cv2_imshow = lambda x: cv2.imshow("window", x)

def convert_image_path(image_path):

    def preprocess_image(image, height):
        ##Resizing the image for smaller pixel sizes
        image=imutils.resize(image,height=height)
        ## Step 1: Converting to gray scale
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ## Step 1.1 : Histogram equalization to normalize contrast across the whole image
        gray=cv2.equalizeHist(gray)
        plt.subplot(1,4,1); plt.imshow(gray, cmap='gray'); plt.title("Gray")
        ## Step 2: Gaussing Blurring for noise reduction
        blurred = cv2.GaussianBlur(gray,(7,7),3,cv2.BORDER_REFLECT_101)
        plt.subplot(1,4,2); plt.imshow(blurred, cmap='gray'); plt.title("Blurred")
        ## Step 3: Adaptive Thresholding
        ## https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
        thresh= cv2.adaptiveThreshold(
            blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3
        )
        plt.subplot(1,4,3); plt.imshow(thresh, cmap='gray'); plt.title("Threshold")
        ## Step 4: Invert to make grid lines white
        thresh=cv2.bitwise_not(thresh)

        ## Step 5: Morphological op
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        processed=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

        plt.subplot(1,4,4); plt.imshow(processed, cmap='gray'); plt.title("Edges")
        return image,processed

    def preprocess(image_path,height=800):
        image=cv2.imread(image_path)
        return preprocess_image(image,height)

    print(image_path)
    # Check if the file exists
    if not os.path.exists(image_path):
        print("Error: Image file not found at", image_path)

    original,processed=preprocess(image_path)
    cv2_imshow(original)
    cv2_imshow(processed)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
        rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
        return rect

    def detect_sudoku_grid(processed,original_image):
        contours = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        sudoku_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                sudoku_contour = approx
                break

        if sudoku_contour is None:
            print("❌ Sudoku grid not found.")
            return None, None

        # Order corners
        sudoku_contour = sudoku_contour.reshape(4, 2)
        rect = order_points(sudoku_contour)

        # Draw contour for visualization
        debug_img = original_image.copy()
        cv2.drawContours(debug_img, [sudoku_contour.astype(int)], -1, (0, 255, 0), 3)

        return rect, debug_img

    rect, debug = detect_sudoku_grid(processed, original)
    plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
    plt.title("Detected Grid")
    plt.axis('off')
    plt.show()

    def wrap_sudoku(order_points,original_image):
        output_size=450
        dst = np.array([
                [0, 0],
                [output_size - 1, 0],
                [output_size - 1, output_size - 1],
                [0, output_size - 1]
            ], dtype="float32")
        M=cv2.getPerspectiveTransform(rect,dst)
        warped=cv2.warpPerspective(original_image,M,(output_size,output_size))
        return warped

    warped_image=wrap_sudoku(rect,original)
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped Sudoku Grid")
    plt.axis("off")
    plt.show()

    def preprocess_wrapped_sudoku(warped, show_steps=False):
        """
        Preprocess a warped Sudoku image for digit extraction.

        Steps:
        1. Convert to grayscale
        2. Histogram equalization
        3. Gaussian blur
        4. Adaptive thresholding (binary inverse)

        Parameters:
            warped (np.ndarray): Warped Sudoku image (top-down view)
            show_steps (bool): If True, displays intermediate steps

        Returns:
            thresh (np.ndarray): Final preprocessed (binary) image
            steps (dict): Dictionary of intermediate results
        """

        steps = {}

        # 1️⃣ Grayscale conversion
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        steps["gray"] = gray

        # 2️⃣ Histogram equalization for better contrast
        gray_eq = cv2.equalizeHist(gray)
        steps["equalized"] = gray_eq

        # 3️⃣ Gaussian blur to reduce small noise
        blurred = cv2.GaussianBlur(gray,(7,7),3,cv2.BORDER_REFLECT_101)
        steps["blurred"] = blurred

        # 4️⃣ Adaptive threshold (invert so digits are white)
        thresh = cv2.adaptiveThreshold(
        blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1
        )
        steps["thresh"] = thresh

        # Optional visualization
        if show_steps:
            titles = ["Gray", "Equalized", "Blurred", "Thresholded"]
            images = [steps[k] for k in ["gray", "equalized", "blurred", "thresh"]]
            for title, img in zip(titles, images):
                print(f" {title}")
                cv2_imshow(img)  # ✅ Show image in Colab

        return thresh, steps


    def split_into_cells(preprocessed_img, grid_size=9, visualize=False):
        """
        Splits a preprocessed Sudoku image into 81 individual cell images (9x9 grid).

        Parameters:
            preprocessed_img (np.ndarray): Binary (thresholded) top-down Sudoku image
            grid_size (int): Number of rows and columns (default 9)
            visualize (bool): If True, shows each cell in a grid visualization

        Returns:
            cells (list): List of 81 cell images (each as np.ndarray)
        """

        h, w = preprocessed_img.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size

        cells = []

        for i in range(grid_size):
            for j in range(grid_size):
                # Crop each cell
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w

                cell = preprocessed_img[y1:y2, x1:x2]

                # Optional cleanup (remove extra border pixels)
                margin = 2
                cell = cell[margin:-margin, margin:-margin] if cell.shape[0] > 4 and cell.shape[1] > 4 else cell

                cells.append(cell)

        # ✅ Visualization grid (only if requested)
        if visualize:
            cell_grid = np.zeros((h, w), dtype=np.uint8)
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w
                    cell_resized = cv2.resize(cells[idx], (cell_w, cell_h))
                    cell_grid[y1:y2, x1:x2] = cell_resized
                    idx += 1
            cv2_imshow(cell_grid)  # ✅ works in Colab

        return cells


    # Suppose `warped` is your top-down Sudoku image
    thresh, steps = preprocess_wrapped_sudoku(warped_image, show_steps=True)

    # Now split 'thresh' into 81 cells
    cells = split_into_cells(thresh,visualize=True)
    print(f"Extracted {len(cells)} cells")

    # Example: visualize one cell
    for cell in cells:
        cv2_imshow(cell)

    
    def detect_digit_debug_with_median_and_close(cell, idx=None):
        if len(cell.shape) == 3:
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            cell_gray = cell

        # --- STEP 1: Median Blur (Applied to Grayscale Image) ---
        # Removes salt-and-pepper noise before thresholding
        cell_blurred = cv2.medianBlur(cell_gray, 5) # 5x5 kernel
        # --- END OF STEP 1 ---

        # Threshold the blurred image
        _, thresh = cv2.threshold(cell_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # --- STEP 2: Morphological Closing (Applied to Binary Image) ---
        # This will fill small *black holes* inside the white digit
        
        # 1. Define a kernel
        kernel = np.ones((3,3), np.uint8)
        
        # 2. Apply the 'MORPH_CLOSE' operation
        thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # --- END OF STEP 2 ---

        # We now find contours on the *cleaned* threshold image
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print(f"[{idx}] No contours")

        h, w = thresh_cleaned.shape
        detected = False
        MIN_AREA_THRESHOLD = 100 # Your noise filter

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA_THRESHOLD:
                continue
                
            x, y, cw, ch = cv2.boundingRect(c)
            aspect_ratio = cw / ch if ch > 0 else 0
            fill_ratio = area / (cw * ch)
            cx, cy = x + cw // 2, y + ch // 2

            print(f"[{idx}] area={area:.1f}, ar={aspect_ratio:.2f}, fill={fill_ratio:.2f}, pos=({cx},{cy})")

            # Visualize the *cleaned* threshold image
            vis = cv2.cvtColor(thresh_cleaned, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 255, 0), 1)
            cv2_imshow(vis)

        vis = cv2.cvtColor(thresh_cleaned, cv2.COLOR_GRAY2BGR)
        return detected, vis

    cells_new = []
    for idx, cell in enumerate(cells):
        detected, celly = detect_digit_debug_with_median_and_close(cell, idx)
        cells_new.append(celly)

    def is_digit_present(cell, idx=None):
        """
        Analyzes a cell and returns True if a digit is found,
        otherwise returns False.
        """
        if len(cell.shape) == 3:
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            cell_gray = cell

        h, w = cell_gray.shape

        # --- Preprocessing (as you had before) ---
        cell_blurred = cv2.medianBlur(cell_gray, 5)
        _, thresh = cv2.threshold(cell_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False # No contours found at all

        # --- Filtering Logic ---
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            # 1. Filter by Area
            # You must tune these min/max values by debugging
            MIN_AREA = 150
            MAX_AREA = (w * h) * 0.7  # Don't allow a contour > 70% of the cell
            if not (MIN_AREA < area < MAX_AREA):
                continue  # Not a digit, try next contour
                
            # 2. Filter by Aspect Ratio (width / height)
            aspect_ratio = cw / ch if ch > 0 else 0
            if not (0.1 < aspect_ratio < 1.5):
                continue  # Too skinny or too wide, probably a grid line
                
            # 3. Filter by Position (is it centered?)
            # Check if the contour's center (cx, cy) is in the
            # middle 80% of the cell.
            cx = x + cw // 2
            cy = y + ch // 2
            if not ((w * 0.1 < cx < w * 0.9) and (h * 0.1 < cy < h * 0.9)):
                continue # It's touching the extreme edge

            # --- If it passed all filters, it's a digit! ---
            # We can stop and return True immediately.
            
            # (Optional: print for final check)
            # print(f"[{idx}] FOUND DIGIT: area={area:.1f}, ar={aspect_ratio:.2f}")
            
            return True

        # If the loop finishes without finding any valid contour...
        return False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 15

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # CNN Model
    class DigitRecognizer(nn.Module):
        def __init__(self):
            super(DigitRecognizer, self).__init__()
            self.conv_layers = nn.Sequential(
                # First conv block
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                # Second conv block
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                
                # Third conv block
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Dropout(0.25)
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.classifier(x)
            return x

    # Load your trained PyTorch model
    model = DigitRecognizer()
    model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/digit_cnn_cuda.pth'))
    model.to(device)
    model.eval()  # Set to evaluation mode

    def preprocessing(cell):
        """Enhanced preprocessing for real-world digit images"""
        if len(cell.shape) == 3:
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            cell_gray = cell
        
        # Resize
        img = cv2.resize(cell_gray, (28, 28))
        
        # Apply aggressive thresholding to get clean binary image
        _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (MNIST expects white digit on black background)
        if np.mean(img_binary) > 127:  # If mostly white
            img_binary = 255 - img_binary
        
        # Center the digit using moments (like MNIST)
        img_centered = center_digit(img_binary)
        
        # Normalize like MNIST training data
        img_normalized = img_centered.astype('float32') / 255.0
        img_normalized = (img_normalized - 0.1307) / 0.3081
        
        # Add batch and channel dimensions for PyTorch
        img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(device)

    def center_digit(img):
        """Center the digit in the 28x28 image using moments"""
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Get largest contour (the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rect
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract digit region
        digit_region = img[y:y+h, x:x+w]
        
        # Calculate padding to center
        pad_x = (28 - w) // 2
        pad_y = (28 - h) // 2
        
        # Create centered image
        centered = np.zeros((28, 28), dtype=np.uint8)
        centered[pad_y:pad_y+h, pad_x:pad_x+w] = digit_region
        
        return centered

    grid = [[0 for _ in range(9)] for _ in range(9)]

    # Your original loop adapted for PyTorch
    for i, cell in enumerate(cells_new):
        if is_digit_present(cell):
            # Preprocess for PyTorch
            img_tensor = preprocessing(cell)
            
            # Show the image (optional)
            img_display = cv2.resize(cell, (28, 28))
            if len(img_display.shape) == 3:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
            cv2_imshow(img_display)
            
            # Prediction
            with torch.no_grad():
                output = model(img_tensor)
                prediction = torch.softmax(output, dim=1)
                predicted_digit = torch.argmax(prediction, dim=1).item()
                confidence = torch.max(prediction).item()

            grid[i // 9][i % 9] = predicted_digit
            
            print(f"Cell {i}: Predicted Digit = {predicted_digit} (Confidence: {confidence:.2f})")

    return grid


image_path = os.path.dirname(os.path.abspath(__file__)) + "\\Sudoku_Dataset\\sudoku2.jpg"
g = convert_image_path(image_path)
print(*g, sep='\n')