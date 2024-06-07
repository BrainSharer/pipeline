import json

with open('annotations.json', 'r') as annotations_file:
    annotation_dict = json.loads(annotations_file.read())
    annotations = annotation_dict['annotations']
    volumes = 0
    polygons = 0
    lines = 0
    for i, annotation in enumerate(annotations):
        print(i)
        for k, v in annotation.items():
            print(f'{k}: {v}')
            if k == 'type' and v == 'volume':
                volumes += 1
            elif k == 'type' and v == 'polygon':
                polygons += 1
            elif k == 'type' and v == 'line':
                lines += 1
        print()
    print(f'Volumes: {volumes}')
    print(f'Polygons: {polygons}')
    print(f'Lines: {lines}')