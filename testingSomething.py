

import shutil
import os

# Sample data (you can replace this with reading from a file)
data = """기타레이어드 | 48240 | ../hair_images/0003.기타레이어드/3638.JS856189/JS856189-085.jpg
원랭스 | 34040 | ../hair_images/0025.원랭스/0316.CP138918/CP138918-009.jpg
허쉬 | 34040 | ../hair_images/0030.허쉬/2979.JS504153/JS504153-116.jpg
에어 | 29418 | ../hair_images/0023.에어/4362.JSS966164/JSS966164-003.jpg
빌드 | 23617 | ../hair_images/0015.빌드/0730.CP472654/CP472654-017.jpg
보브 | 20811 | ../hair_images/0014.보브/1736.DSS452948/DSS452948-087.jpg
바디 | 20810 | ../hair_images/0011.바디/1698.DSS404803/DSS404803-099.JPG
보니 | 15469 | ../hair_images/0013.보니/2272.JS046676/JS046676-171.jpg
플리츠 | 15370 | ../hair_images/0029.플리츠/1456.DSS106604/DSS106604-091.jpg
숏단발 | 13614 | ../hair_images/0017.숏단발/2837.JS426948/JS426948-144.jpg
히피 | 13455 | ../hair_images/0031.히피/0594.CP346968/CP346968-023.jpg
남자일반숏 | 12577 | ../hair_images/0005.남자일반숏/0906.CP606502/CP606502-013.jpg
가르마 | 12510 | ../hair_images/0001.가르마/1635.DSS306132/DSS306132-101.jpg
기타남자스타일 | 11242 | ../hair_images/0002.기타남자스타일/3263.JS644904/JS644904-074.jpg
쉐도우 | 10298 | ../hair_images/0018.쉐도우/3118.JS568488/JS568488-075.jpg
리젠트 | 9837 | ../hair_images/0008.리젠트/4146.JSS512878/JSS512878-055.jpg
애즈 | 8143 | ../hair_images/0022.애즈/3310.JS668799/JS668799-035.jpg
포마드 | 7970 | ../hair_images/0028.포마드/3146.JS584639/JS584639-012.jpg
댄디 | 7798 | ../hair_images/0006.댄디/3769.JS970808/JS970808-049.jpg
소프트투블럭댄디 | 7446 | ../hair_images/0016.소프트투블럭댄디/0770.CP498058/CP498058-036.jpg
리프 | 7276 | ../hair_images/0009.리프/2832.JS424606/JS424606-085.jpg
미스티 | 6161 | ../hair_images/0010.미스티/0157.CP046513/CP046513-032.jpg
여자일반숏 | 5527 | ../hair_images/0024.여자일반숏/2853.JS434891/JS434891-021.jpg
시스루댄디 | 4981 | ../hair_images/0021.시스루댄디/2174.JS002514/JS002514-158.jpg
스핀스왈로 | 4936 | ../hair_images/0020.스핀스왈로/3292.JS656553/JS656553-057.jpg
원블럭댄디 | 3869 | ../hair_images/0026.원블럭댄디/0658.CP400927/CP400927-009.jpg
베이비 | 3076 | ../hair_images/0012.베이비/2746.JS380846/JS380846-043.jpg
기타여자스타일 | 2964 | ../hair_images/0004.기타여자스타일/2849.JS432735/JS432735-095.jpg
테슬 | 2926 | ../hair_images/0027.테슬/4470.MN108454/MN108454-033.jpg
루프 | 2544 | ../hair_images/0007.루프/3104.JS558513/JS558513-052.jpg
쉼표 | 2401 | ../hair_images/0019.쉼표/3649.JS864811/JS864811-083.jpg
"""
copied_directory = "./sample_images/"

# Process each line
for line in data.strip().split('\n'):
    parts = [part.strip() for part in line.split('|')]
    if len(parts) != 3:
        continue
    name, _, src_path = parts
    # Get the extension from the source file
    ext = os.path.splitext(src_path)[1]
    dst_path = copied_directory + (f"{name}{ext}")

    try:
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")
    except FileNotFoundError:
        print(f"File not found: {src_path}")
    except Exception as e:
        print(f"Error copying {src_path}: {e}")
