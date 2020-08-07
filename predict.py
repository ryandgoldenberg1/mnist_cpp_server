import json
import requests
import matplotlib.pyplot as plt


# Note: You must be running the server locally for this to work.
#       For instructions on how to run it, see the README.
def main():
    # Load the image data
    images = []
    for i in range(1, 6):
        with open(f'data/example_{i}.json') as f:
            images.append(json.load(f))

    # Make the requests to the local server for predictions
    predictions = []
    for data in images:
        response = requests.post('http://localhost:3000/predict', data=json.dumps(data))
        predictions.append(response.text)

    fig, axs = plt.subplots(1, 5, figsize=(15, 3), tight_layout=True)
    fig.subplots_adjust(top=0.8)
    for i, ax in enumerate(axs):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(predictions[i], pad=30, fontsize=42)
    plt.show()


if __name__ == '__main__':
    main()
