from PIL import Image
import os
import argparse
import sys


class ImageProcessor:
    def __init__(self, source_path: str, dest_path: str):
        """
        Image processor resizes source jpeg files to have the smallest image dimension 512px.
        Destination folder must not exist.

        Args:
            source_path (str): Path to the source folder containing JPG images
            dest_path (str): Path to the destination folder for processed images
        """
        self.source_path = source_path
        self.dest_path = dest_path

        # Check if destination directory exists
        if os.path.exists(dest_path):
            print(f"Error: Destination folder '{dest_path}' already exists. Please specify a new destination.",
                  file=sys.stderr)
            sys.exit(1)

        try:
            # Create destination directory
            os.makedirs(dest_path)
        except PermissionError:
            print(f"Error: Permission denied when creating directory '{dest_path}'",
                  file=sys.stderr)
            sys.exit(1)
        except OSError as e:
            print(f"Error creating directory '{dest_path}': {str(e)}",
                  file=sys.stderr)
            sys.exit(1)

    def process_images(self):
        """Process all JPG images in the source folder."""
        # Get all jpg files from source directory
        jpg_files = [f for f in os.listdir(self.source_path)
                     if f.lower().endswith(('.jpg', '.jpeg'))]

        processed = 0
        failed = 0
        for filename in jpg_files:
            if self._process_single_image(filename):
                processed += 1
            else:
                failed += 1

        print(f"\nProcessing complete:")
        print(f"Successfully processed: {processed} images")
        print(f"Failed to process: {failed} images")

    def _process_single_image(self, filename: str) -> bool:
        """
        Process a single image: resize to 512px smallest dimension.

        Args:
            filename (str): Name of the image file to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        input_path = os.path.join(self.source_path, filename)
        try:
            # Open image
            image = Image.open(input_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize image so smallest dimension is 512px
            width, height = image.size
            if width < height:
                new_width = 512
                new_height = int(height * (512 / width))
            else:
                new_height = 512
                new_width = int(width * (512 / height))

            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save processed image
            output_path = os.path.join(self.dest_path, filename)
            resized_image.save(output_path, 'JPEG', quality=95)
            return True

        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            print(f"Error processing '{filename}': {str(e)}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Unexpected error processing '{filename}': {str(e)}", file=sys.stderr)
            return False


def parse_args():
    parser = argparse.ArgumentParser(description='Resize images to 512px on smallest dimension')
    parser.add_argument('source', type=str, help='Source folder containing JPG images', default='../content/style1000')
    parser.add_argument('destination', type=str, help='Destination folder for processed images',
                        default='../content/style1000resized')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check if source folder exists
    if not os.path.exists(args.source):
        print(f"Error: Source folder '{args.source}' does not exist.",
              file=sys.stderr)
        sys.exit(1)

    # Check if source is a directory
    if not os.path.isdir(args.source):
        print(f"Error: '{args.source}' is not a directory.",
              file=sys.stderr)
        sys.exit(1)

    processor = ImageProcessor(args.source, args.destination)
    processor.process_images()