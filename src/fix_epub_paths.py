"""
Script to fix image paths in EPUBs that have incorrect relative paths.

Usage: python fix_epub_image_paths.py input.epub output.epub
"""

import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from lxml import etree
import os


def fix_epub_image_paths(input_path: str, output_path: str):
    """Fix image paths in EPUB files."""
    
    print(f"Processing: {input_path}")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract EPUB
        print("Extracting EPUB...")
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        
        # Find all XHTML files
        xhtml_files = list(temp_path.rglob("*.xhtml")) + list(temp_path.rglob("*.html"))
        print(f"Found {len(xhtml_files)} XHTML files")
        
        # Process each XHTML file
        fixed_count = 0
        for xhtml_file in xhtml_files:
            # Get relative path from EPUB root
            try:
                relative_path = xhtml_file.relative_to(temp_path / "EPUB")
                epub_relative = True
            except ValueError:
                relative_path = xhtml_file.relative_to(temp_path)
                epub_relative = False
            
            print(f"\nProcessing: {relative_path}")
            
            # Parse XHTML
            parser = etree.XMLParser(recover=True, encoding='utf-8')
            try:
                tree = etree.parse(str(xhtml_file), parser)
                
                # Find all img tags
                # Try with namespace first
                nsmap = {'html': 'http://www.w3.org/1999/xhtml'}
                images = tree.xpath('//html:img', namespaces=nsmap)
                
                if not images:
                    # Try without namespace
                    images = tree.xpath('//img')
                
                for img in images:
                    src = img.get('src')
                    if src and 'images/' in src:
                        # Determine correct path
                        if '/' in str(relative_path.parent) and relative_path.parent != Path('.'):
                            # XHTML is in subdirectory
                            if not src.startswith('../'):
                                new_src = f"../{src.lstrip('./')}"
                                print(f"  Fixed: {src} -> {new_src}")
                                img.set('src', new_src)
                                fixed_count += 1
                        else:
                            # XHTML is in root
                            if src.startswith('../'):
                                new_src = src.replace('../', '')
                                print(f"  Fixed: {src} -> {new_src}")
                                img.set('src', new_src)
                                fixed_count += 1
                
                # Save updated XHTML
                tree.write(str(xhtml_file), 
                          encoding='utf-8', 
                          xml_declaration=True,
                          pretty_print=True)
                
            except Exception as e:
                print(f"  Error processing {xhtml_file}: {e}")
        
        print(f"\nFixed {fixed_count} image paths")
        
        # Repackage EPUB
        print("Repackaging EPUB...")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for file_path in temp_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_path)
                    zip_ref.write(file_path, arcname)
        
        print(f"✅ Fixed EPUB saved to: {output_path}")


def check_epub_structure(epub_path: str):
    """Check and display EPUB structure for debugging."""
    print(f"\nAnalyzing EPUB structure: {epub_path}")
    
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        
        # Find XHTML files
        xhtml_files = [f for f in files if f.endswith('.xhtml') or f.endswith('.html')]
        image_files = [f for f in files if 'images/' in f]
        
        print(f"\nXHTML files ({len(xhtml_files)}):")
        for f in xhtml_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(xhtml_files) > 5:
            print(f"  ... and {len(xhtml_files) - 5} more")
        
        print(f"\nImage files ({len(image_files)}):")
        for f in image_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more")
        
        # Check structure
        if xhtml_files:
            first_xhtml = xhtml_files[0]
            if '/' in first_xhtml and not first_xhtml.startswith('EPUB/'):
                print("\n⚠️  XHTMLs are in subdirectories - images need '../' prefix")
            else:
                print("\n✓ XHTMLs are in EPUB root - images need direct path")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_epub_image_paths.py input.epub output.epub")
        print("\nThis script fixes incorrect image paths in EPUB files.")
        print("It automatically detects the EPUB structure and adjusts paths accordingly.")
        sys.exit(1)
    
    input_epub = sys.argv[1]
    output_epub = sys.argv[2]
    
    if not os.path.exists(input_epub):
        print(f"Error: Input file not found: {input_epub}")
        sys.exit(1)
    
    # First check the structure
    check_epub_structure(input_epub)
    
    # Then fix the paths
    fix_epub_image_paths(input_epub, output_epub)
