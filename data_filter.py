import os
import shutil

def count_dicom_files(folder_path):
    """
    Count the number of DICOM files in a folder.
    DICOM files typically have .dcm extension or no extension.
    """
    dicom_count = 0
    try:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                # Check for .dcm files or files without extension (common for DICOM)
                # Exclude .xml files
                if (file.endswith('.dcm') or ('.' not in file)) and not file.endswith('.xml'):
                    dicom_count += 1
    except Exception as e:
        print(f"Error counting files in {folder_path}: {str(e)}")
    
    return dicom_count

def organize_folders_by_dicom_count(base_folder, min_dicom_files=100):
    """
    Organize subfolders into data_filtered/less_than_100/ and data_filtered/more_than_100/
    based on DICOM file count.
    
    Parameters:
    - base_folder: root folder to start scanning
    - min_dicom_files: threshold for categorization (default: 100)
    """
    
    if not os.path.exists(base_folder):
        print(f"Error: Base folder '{base_folder}' does not exist!")
        return
    
    # Create output directories
    output_base = "data_filtered"
    less_than_dir = os.path.join(output_base, "less_than_100")
    more_than_dir = os.path.join(output_base, "more_than_100")
    
    os.makedirs(less_than_dir, exist_ok=True)
    os.makedirs(more_than_dir, exist_ok=True)
    
    print(f"Starting scan of: {base_folder}")
    print(f"Threshold: {min_dicom_files} DICOM files")
    print(f"Output directory: {output_base}/")
    print("="*60)
    
    less_than_count = 0
    more_than_count = 0
    error_count = 0
    
    # Walk through all directories and find leaf folders (folders with DICOM files)
    print("\nScanning and moving subfolders...")
    for root, dirs, files in os.walk(base_folder, topdown=True):
        # Skip the base folder itself
        if root == base_folder:
            continue
        
        # Count DICOM files in current directory
        dicom_count = count_dicom_files(root)
        
        # Only process folders that actually contain DICOM files (leaf folders)
        if dicom_count > 0:
            # Determine destination based on DICOM count
            if dicom_count < min_dicom_files:
                destination_base = less_than_dir
                category = "< 100"
                less_than_count += 1
            else:
                destination_base = more_than_dir
                category = ">= 100"
                more_than_count += 1
            
            # Create a unique folder name to avoid conflicts
            # Use relative path structure to preserve hierarchy
            rel_path = os.path.relpath(root, base_folder)
            destination = os.path.join(destination_base, rel_path)
            
            try:
                # Move the folder
                print(f"Moving: {root} ({dicom_count} DICOM files) -> {category}")
                shutil.move(root, destination)
            except Exception as e:
                error_count += 1
                print(f"Error moving {root}: {str(e)}")
    
    # Clean up empty parent folders in base directory
    print("\n" + "="*60)
    print("Cleaning up empty parent folders...")
    empty_folders_deleted = 0
    
    for root, dirs, files in os.walk(base_folder, topdown=False):
        # Skip the base folder itself
        if root == base_folder:
            continue
        
        try:
            # Check if folder is empty
            if not os.listdir(root):
                print(f"Deleting empty folder: {root}")
                os.rmdir(root)
                empty_folders_deleted += 1
        except Exception as e:
            print(f"Error deleting empty folder {root}: {str(e)}")
    
    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  - Moved to 'less_than_100/': {less_than_count} folders")
    print(f"  - Moved to 'more_than_100/': {more_than_count} folders")
    print(f"  - Empty parent folders deleted: {empty_folders_deleted}")
    print(f"  - Errors encountered: {error_count}")
    print(f"  - Total folders processed: {less_than_count + more_than_count}")
    print("="*60)
    print(f"\nOrganized data is available in: {output_base}/")

def create_flat_dataset(source_folder, output_folder="data"):
    """
    Create a flat dataset structure where each DICOM directory contains
    DICOM files and corresponding XML file.
    
    Structure: data/<unique_folder_name>/<dicom_files + xml_file>
    
    Parameters:
    - source_folder: folder containing organized data (e.g., data_filtered/more_than_100)
    - output_folder: output directory name (default: 'data')
    """
    
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist!")
        return
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Creating flat dataset from: {source_folder}")
    print(f"Output directory: {output_folder}/")
    print("="*60)
    
    processed_count = 0
    error_count = 0
    
    # Walk through source folder
    for root, dirs, files in os.walk(source_folder):
        # Check if this folder contains DICOM files
        dicom_count = count_dicom_files(root)
        
        if dicom_count > 0:
            # Find XML file in the folder
            xml_files = [f for f in files if f.endswith('.xml')]
            
            if not xml_files:
                print(f"Warning: No XML file found in {root}")
                continue
            
            # Create a unique folder name
            # Use the parent folder names to create unique identifier
            rel_path = os.path.relpath(root, source_folder)
            folder_name = rel_path.replace(os.sep, '_')
            
            destination = os.path.join(output_folder, folder_name)
            
            try:
                # Copy the entire folder (DICOM + XML files)
                if os.path.exists(destination):
                    print(f"Skipping (already exists): {destination}")
                else:
                    shutil.copytree(root, destination)
                    processed_count += 1
                    print(f"Copied: {root} -> {destination}")
                    print(f"  DICOM files: {dicom_count}, XML files: {len(xml_files)}")
            except Exception as e:
                error_count += 1
                print(f"Error copying {root}: {str(e)}")
    
    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  - Folders processed: {processed_count}")
    print(f"  - Errors encountered: {error_count}")
    print("="*60)
    print(f"\nFlat dataset created in: {output_folder}/")

def display_menu():
    """Display the menu options"""
    print("\n" + "="*60)
    print("DICOM Data Filtering & Organization Tool")
    print("="*60)
    print("1. Filter and organize folders by DICOM count")
    print("   (Creates data_filtered/less_than_100/ and data_filtered/more_than_100/)")
    print("\n2. Create flat dataset structure")
    print("   (Creates data/ with flat folder structure)")
    print("\n3. Both (Filter + Create flat dataset)")
    print("\n4. Exit")
    print("="*60)

def main():
    """Main menu-driven program"""
    
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Filter and organize
            print("\n--- Filter and Organize ---")
            base_folder = input("Enter base folder path (default: manifest-1600709154662/LIDC-IDRI): ").strip()
            if not base_folder:
                base_folder = "manifest-1600709154662/LIDC-IDRI"
            
            min_files = input("Enter minimum DICOM files threshold (default: 100): ").strip()
            if not min_files:
                min_files = 100
            else:
                try:
                    min_files = int(min_files)
                except:
                    print("Invalid input, using default: 100")
                    min_files = 100
            
            print(f"\n⚠️  This will move folders from: {base_folder}")
            print(f"   to: data_filtered/less_than_100/ and data_filtered/more_than_100/")
            confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                organize_folders_by_dicom_count(base_folder, min_files)
            else:
                print("Operation cancelled.")
        
        elif choice == '2':
            # Create flat dataset
            print("\n--- Create Flat Dataset ---")
            source_folder = input("Enter source folder path (default: data_filtered/more_than_100): ").strip()
            if not source_folder:
                source_folder = "data_filtered/more_than_100"
            
            output_folder = input("Enter output folder name (default: data): ").strip()
            if not output_folder:
                output_folder = "data"
            
            print(f"\n⚠️  This will copy folders from: {source_folder}")
            print(f"   to: {output_folder}/")
            confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                create_flat_dataset(source_folder, output_folder)
            else:
                print("Operation cancelled.")
        
        elif choice == '3':
            # Both operations
            print("\n--- Filter + Create Flat Dataset ---")
            base_folder = input("Enter base folder path (default: manifest-1600709154662/LIDC-IDRI): ").strip()
            if not base_folder:
                base_folder = "manifest-1600709154662/LIDC-IDRI"
            
            min_files = input("Enter minimum DICOM files threshold (default: 100): ").strip()
            if not min_files:
                min_files = 100
            else:
                try:
                    min_files = int(min_files)
                except:
                    print("Invalid input, using default: 100")
                    min_files = 100
            
            output_folder = input("Enter final output folder name (default: data): ").strip()
            if not output_folder:
                output_folder = "data"
            
            print(f"\n⚠️  This will:")
            print(f"   1. Filter and move folders from: {base_folder}")
            print(f"   2. Create flat dataset in: {output_folder}/")
            confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                # Step 1: Filter
                organize_folders_by_dicom_count(base_folder, min_files)
                
                # Step 2: Create flat dataset
                print("\n" + "="*60)
                print("Starting flat dataset creation...")
                print("="*60)
                create_flat_dataset("data_filtered/more_than_100", output_folder)
            else:
                print("Operation cancelled.")
        
        elif choice == '4':
            print("\nExiting program. Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        # Ask if user wants to continue
        continue_choice = input("\nPress Enter to continue or type 'exit' to quit: ").strip().lower()
        if continue_choice == 'exit':
            print("\nExiting program. Goodbye!")
            break

if __name__ == "__main__":
    main()