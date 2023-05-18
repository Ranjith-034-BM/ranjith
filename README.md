# ranjith
surface area calculation methods 
# trapezoidal rule

def trapezoidal():
    import cv2
    import numpy as np
    from scipy.integrate import trapz

    # Load the image and convert it to grayscale
    image = cv2.imread('leather_piece.jpeg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the perimeter of the contour using arcLength
    perimeter = cv2.arcLength(largest_contour, True)

    # Get the x and y coordinates of the contour
    x_coords = largest_contour[:, 0, 0]
    y_coords = largest_contour[:, 0, 1]

    # Perform trapezoidal numerical integration to calculate the surface area
    surface_area = trapz(y_coords, x_coords)

    print("Surface Area:", surface_area)

def simpsons():
    import cv2
    import numpy as np
    from scipy.integrate import simps

    # Load the image and convert it to grayscale
    image = cv2.imread('leather_piece.jpeg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Sort the contour points based on their x-coordinates
    contour_points = largest_contour.reshape(-1, 2)
    contour_points = contour_points[np.argsort(contour_points[:, 0])]

    # Get the x and y coordinates of the contour
    x_coords = contour_points[:, 0]
    y_coords = contour_points[:, 1]

    # Perform Simpson's 1/3 rule numerical integration to calculate the surface area
    surface_area = simps(y_coords, x_coords, even='first')

    print("Surface Area:", surface_area)

def pick():
    import cv2
    import numpy as np

    # Load the image and convert it to grayscale
    image = cv2.imread('leather_piece.jpeg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Flatten the contour coordinates
    contour_points = largest_contour.reshape(-1, 2)

    # Compute the signed area using Pick's method
    signed_area = 0
    num_points = len(contour_points)

    for i in range(num_points):
        x1, y1 = contour_points[i]
        x2, y2 = contour_points[(i + 1) % num_points]
        signed_area += (x1 * y2) - (x2 * y1)

    # Calculate the absolute value of the area
    surface_area = abs(signed_area) / 2

    print("Surface Area:", surface_area)

def boundaryapp():
    import cv2
    import numpy as np

    # Load the image and convert it to grayscale
    image = cv2.imread('leather_piece.jpeg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour with a simpler polygon
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Calculate the surface area of the approximated contour
    surface_area = cv2.contourArea(approximated_contour)

    print("Surface Area:", surface_area)
trapezoidal()
pick()
boundaryapp()


from tkinter import *

root = Tk()
root.geometry('600x600')
root.title('surface area calculator')

trap_button = Button(root,text = "try Trapezoidal rule", command = trapezoidal)
trap_button.pack()

pick_button = Button(root,text = "try pick's method" , command = pick)
pick_button.pack()

bound_button = Button(root,text = "try Boundary Approximation",command = boundaryapp)
bound_button.pack()

root.mainloop()

