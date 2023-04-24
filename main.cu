#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for image resizing
__global__ void resizeImage(const unsigned char* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX < dstWidth && dstY < dstHeight)
    {
        float srcX = static_cast<float>(dstX) / dstWidth * srcWidth;
        float srcY = static_cast<float>(dstY) / dstHeight * srcHeight;

        int srcX1 = static_cast<int>(srcX);
        int srcY1 = static_cast<int>(srcY);

        int srcIndex = srcY1 * srcWidth + srcX1;
        int dstIndex = dstY * dstWidth + dstX;

        dst[dstIndex] = src[srcIndex];
    }
}

int main()
{
    // Load PGM image
    const char* filename = "input2.pgm";
    FILE* file = fopen(filename, "rb");
    if (!file)
    {
        printf("Failed to open file: %s\n", filename);
        return 1;
    }

    char header[3];
    int width, height, maxVal;
    fread(header, 1, 3, file);
    fscanf(file, "%d %d %d", &width, &height, &maxVal);

    int imageSize = width * height;
    unsigned char* h_srcImage = new unsigned char[imageSize];
    fread(h_srcImage, 1, imageSize, file);
    fclose(file);

    // Allocate memory on GPU
    unsigned char* d_srcImage, * d_dstImage;
    cudaMalloc(&d_srcImage, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_dstImage, imageSize * sizeof(unsigned char));

    // Copy input image from host to device
    cudaMemcpy(d_srcImage, h_srcImage, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define grid and block sizes for CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Call CUDA kernel for image resizing
    resizeImage<<<gridSize, blockSize>>>(d_srcImage, d_dstImage, width, height, width / 2, height / 2);

    // Copy output image from device to host
    unsigned char* h_dstImage = new unsigned char[imageSize / 4];
    cudaMemcpy(h_dstImage, d_dstImage, imageSize / 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save resized image to file
    const char* outFilename = "output2.pgm";
    FILE* outFile = fopen(outFilename, "wb");
    if (!outFile)
    {
        printf("Failed to create output file: %s\n", outFilename);
        return 1;
    }

    fprintf(outFile, "P5\n%d %d\n%d\n", width / 2, height / 2, maxVal);
    fwrite(h_dstImage, 1, imageSize / 4, outFile);
    fclose(outFile);

    // Clean up
    delete[] h_srcImage;
    delete[] h_dstImage;
    cudaFree(d_srcImage);
    cudaFree(d_dstImage);

    return 0;
}
