import os
import pydicom
from pydicom.multival import MultiValue
import argparse
import subprocess

# Devuelve el top x de carpetas con más archivos


def top_folders_by_file_count(root_folder, amount):
    folder_file_count = {}

    with os.scandir(root_folder) as it:
        for entry in it:
            if entry.is_dir():
                folder_file_count[entry.name] = len(os.listdir(entry.path))

    # Ordena las carpetas por la cantidad de archivos de mayor a menor y selecciona las top 'amount'
    top_folders = sorted(
        folder_file_count, key=folder_file_count.get, reverse=True)[:amount]

    return top_folders

# Obtiene los valores Series Description y Window Width de la metadata


def get_metadata(folder_path):
    try:
        first_file_path = next(os.scandir(folder_path)).path
        # Leer metadata del primer archivo
        metadata = pydicom.dcmread(first_file_path)
        # Acceder a los atributos específicos
        series_description = metadata.get('SeriesDescription', "No disponible")
        window_width = metadata.get('WindowWidth', "No disponible")
    except Exception:
        return "No disponible", "No disponible"

    return series_description, window_width

# Determina si la serie contiene la palabra "bone" o "hueso"


def is_contain_bone(series_description):
    return "bone" in series_description.lower() or "hueso" in series_description.lower()

# Evalúa si window_width es >= 1500


def is_between_values(window_width):
    if isinstance(window_width, MultiValue):
        return any(width >= 1500 for width in window_width)
    return isinstance(window_width, (int, float)) and window_width >= 1500

# Devuelve el path de la carpeta a utilizar en las mediciones


def get_folder_TC(root_folder):
    top_folders = top_folders_by_file_count(root_folder, 3)

    for folder_name in top_folders:
        folder_path = os.path.join(root_folder, folder_name)
        series_description, window_width = get_metadata(folder_path)

        if is_contain_bone(series_description) or is_between_values(window_width):
            return folder_path
    return None

# Valida si un archivo pertenece a una tomografía
# El valor 1.2.840.10008.1.3.10 solo esta presente en un DICOMDIR, se valida que no tenga ese valor


def is_ct_dicom_file(file_path):
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
        return dcm.file_meta.MediaStorageSOPClassUID != '1.2.840.10008.1.3.10'
    except Exception:
        return False

# Recorre los directorios y subdirectorios, y devuelve el path de la carpeta a utilizar en las mediciones


def get_final_path_CT(directorio):
    for root, _, files in os.walk(directorio):
        for file in files:
            file_path = os.path.join(root, file)
            if is_ct_dicom_file(file_path):
                return get_folder_TC(os.path.dirname(root))
    return "La carpeta seleccionada no cumple con el formato de una tomografía"

# Transforma un .dcm a un .nii.gz


def convert_dcm_to_nii(dicom_folder, output_folder):
    # Verifica que el directorio de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Comando para convertir de DICOM a NIfTI
    command = f'dcm2niix -o {output_folder} -z y -p n -m y -x n {dicom_folder}'

    # Ejecutar el comando
    subprocess.run(command, shell=True, check=True)

    # -z y: compress the output NIfTI file into .nii.gz
    # -o: specify the output directory

# --------------------------------------------------------------------------------------------------

# Función principal


def dcm_to_nii():
    parser = argparse.ArgumentParser(
        description="Script para encontrar el directorio de tomografía a partir de una ruta y luego convertirlo a .nii.gz")
    parser.add_argument(
        "ruta", help="Ruta del directorio raíz para iniciar la búsqueda.")
    parser.add_argument("carpeta_de_salida",
                        help="Ruta a la carpeta donde quedara el .nii.gz")
    args = parser.parse_args()

    selectedPath = get_final_path_CT(args.ruta)
    print(selectedPath)

    dicom_folder = selectedPath
    output_folder = args.carpeta_de_salida
    convert_dcm_to_nii(dicom_folder, output_folder)
