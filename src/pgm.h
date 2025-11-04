#ifndef PGM_H
#define PGM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int width;
    int height;
    int max_gray;
    unsigned char *data;
} PGMImage;

// Leer imagen PGM
PGMImage* readPGM(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: No se pudo abrir el archivo %s\n", filename);
        return NULL;
    }

    PGMImage *img = (PGMImage*)malloc(sizeof(PGMImage));
    char magic[3];
    
    // Leer magic number (P5 para PGM binario)
    fscanf(fp, "%s", magic);
    if (strcmp(magic, "P5") != 0) {
        printf("Error: Formato no soportado (debe ser P5)\n");
        fclose(fp);
        free(img);
        return NULL;
    }

    // Saltar comentarios
    char c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n');
        c = getc(fp);
    }
    ungetc(c, fp);

    // Leer dimensiones y valor máximo
    fscanf(fp, "%d %d %d", &img->width, &img->height, &img->max_gray);
    fgetc(fp); // Saltar newline

    // Leer datos de la imagen
    int size = img->width * img->height;
    img->data = (unsigned char*)malloc(size);
    fread(img->data, 1, size, fp);

    fclose(fp);
    return img;
}

// Escribir imagen PGM
void writePGM(const char *filename, PGMImage *img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: No se pudo crear el archivo %s\n", filename);
        return;
    }

    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max_gray);
    fwrite(img->data, 1, img->width * img->height, fp);

    fclose(fp);
}

// Liberar memoria de la imagen
void freePGM(PGMImage *img) {
    if (img) {
        if (img->data) free(img->data);
        free(img);
    }
}

// Crear imagen PGM vacía
PGMImage* createPGM(int width, int height) {
    PGMImage *img = (PGMImage*)malloc(sizeof(PGMImage));
    img->width = width;
    img->height = height;
    img->max_gray = 255;
    img->data = (unsigned char*)calloc(width * height, 1);
    return img;
}

#endif // PGM_H
