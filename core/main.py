from scanning import *
import cv2 as cv

# Main entry point
if __name__ == "__main__":
    '''
    scanner = CardScanner()
    try:
        scanner.start()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        scanner.stop()
    '''

    image = cv.imread("../assets/mtg-card-underworld-cerberus.jpg")
    processed = ImageUtils.preprocess_for_ocr(image)
    # card = CardDetector.detect_card(image)
    cv.imshow("Magic Lens - Original Frame", image)
    cv.imshow("Magic Lens - Processed Frame (Adaptive Thres)", processed)
    # cv.imshow("Magic Lens - Detected Card", card)
    cv.waitKey(0)
    cv.destroyAllWindows()
