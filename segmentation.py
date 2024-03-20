import numpy as np
import warnings
import time
import cv2
import os

from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt


def preprocess_image(image):
    """
        Predspracuje obrázok pre segmentáciu. Najskôr sa aplikuje histogramová rovnováha, ktorá zvýrazní detaily v
        tmavších oblastiach (transformuje intenzity pixelov tak, aby boli rovnomerne rozložené na celom rozsahu jasu).
        Následne sa aplikuje Gaussovský blur, ktorý rozostrí obrázok.
        Nakoniec sa použije Bilateralfilter na zjemnenie obrazu a zároveň zachovanie hrán a ostrých detailov.

        :return: Predspracovaný obrázok.
        :param image: Obrázok, ktorý sa má spracovať.
    """

    equalized = cv2.equalizeHist(image)
    blurred = cv2.GaussianBlur(equalized, (201, 201), 20)  # Vyššia hodnota sigmaX vedie k väčšiemu rozmazaniu.

    filtered_1 = cv2.bilateralFilter(blurred, d=500, sigmaColor=20, sigmaSpace=250)
    filtered_2 = cv2.bilateralFilter(filtered_1, d=500, sigmaColor=20, sigmaSpace=150)
    filtered_3 = cv2.bilateralFilter(filtered_2, d=500, sigmaColor=20, sigmaSpace=120)
    filtered_4 = cv2.bilateralFilter(filtered_3, d=500, sigmaColor=20, sigmaSpace=100)

    return filtered_4


def apply_gmm(image, n_components):
    """
        Aplikuje Gaussian Mixture Model klastrovanie na obrázok. Najskôr sa obrázok prevedie na 1D pole,
        následne sa na ňom aplikuje Gaussian Mixture klastrovanie. Potom sa výsledok prevedie späť na 2D pole.

        :return: Segmentovaný obrázok s dátovym typom np.uint8.
        :param image: Obrázok, na ktorý sa má aplikovať GMM klastrovanie.
        :param n_components: Počet komponentov pre GMM.
        :type n_components: int
    """

    reshaped = image.reshape((-1, 1))

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", max_iter=200, random_state=42)
    gmm.fit(reshaped)

    segmented = gmm.predict(reshaped).reshape(image.shape)

    return segmented.astype(np.uint8)


def segment_images(input_directory, output_directory, n_components):
    """
        Na začiatku sa vytvorí výstupný priečinok, ak neexistuje. Následne sa načítajú vstupné obrázky.
        Potom sa na každý obrázok aplikuje segmentácia.
        Na konci sa použije morfologická operácia open, ktorá odstráni malé objekty a ostré hrany.
        Vizualizácia originálneho obrázku a segmentovaného obrázku sa zobrazia pre prvé 3 obrázky pomocou knižnice matplotlib.
        Obrázky sa následne uložia do výstupného priečinku.

        :param input_directory: Priečinok obsahujúci vstupné obrázky.
        :param output_directory: Priečinok pre uloženie výsledných segmentovaných obrázkov.
        :param n_components: Počet komponentov pre GMM.
        :type n_components: int
    """

    orig_plot, seg_plot = [], []
    plot_images = ["tm/tm1_1_1.png", "tm/tm2_1_1.png", "tm/tm3_1_1.png"]

    start_time = time.time()

    os.makedirs(output_directory, exist_ok=True)

    valid_formats = {".png", ".jpg", ".jpeg"}
    png_images = [file for file in os.listdir(input_directory) if file.lower().endswith(tuple(valid_formats))]

    with warnings.catch_warnings():  # ignoruje warningy z knižnice scikit-learn - n_jobs
        warnings.simplefilter("ignore", category=FutureWarning)

        for png_image in png_images:
            image_path = os.path.join(input_directory, png_image)
            print(f"Segmenting image: {image_path}")
            image = cv2.imread(image_path, 0)

            preprocessed = preprocess_image(image)
            segmented = apply_gmm(preprocessed, n_components)
            morph = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))  # Kernel size = (30, 30)

            if image_path in plot_images:
                list.append(orig_plot, image)
                list.append(seg_plot, morph)

            output_path = os.path.join(output_directory, png_image)
            cv2.imwrite(output_path, morph)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(orig_plot[0], cv2.COLOR_BGR2RGB))
    plt.title("Original Image 1")

    plt.subplot(1, 6, 2)
    plt.imshow(seg_plot[0], cmap="viridis")
    plt.title("Segmented Image 1")

    plt.subplot(1, 6, 3)
    plt.imshow(cv2.cvtColor(orig_plot[1], cv2.COLOR_BGR2RGB))
    plt.title("Original Image 2")

    plt.subplot(1, 6, 4)
    plt.imshow(seg_plot[1], cmap="viridis")
    plt.title("Segmented Image 2")

    plt.subplot(1, 6, 5)
    plt.imshow(cv2.cvtColor(orig_plot[2], cv2.COLOR_BGR2RGB))
    plt.title("Original Image 3")

    plt.subplot(1, 6, 6)
    plt.imshow(seg_plot[2], cmap="viridis")
    plt.title("Segmented Image 3")

    plt.tight_layout()
    plt.show()

    print(f"\nSegmentation finished. Elapsed time: {(time.time() - start_time) / 60:.2f} minutes.")


def main():

    input_directory = "tm/"
    output_directory = "output/"
    n_components = 5
    segment_images(input_directory, output_directory, n_components)


if __name__ == "__main__":
    main()
