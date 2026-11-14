"""Join explicit patient mappings with the laboratory's headerless slide labels."""
import argparse
import csv
from pathlib import Path
from .data import read_cohort


def prepare(feature_root, mapping, output, feature_format='h5'):
    root=Path(feature_root).resolve(); output=Path(output).resolve()
    with (root/'label.csv').open(newline='') as f:
        labels={}
        for row in csv.reader(f):
            if not row: continue
            if row[:2]==['slide_id','label']: continue
            if len(row)!=2 or row[0] in labels or row[1] not in ('0','1'):
                raise ValueError('label.csv requires unique slide_id,binary_label rows')
            labels[row[0]]=int(row[1])
    with Path(mapping).open(newline='') as f:
        reader=csv.DictReader(f)
        if not {'slide_id','patient_id'}<=set(reader.fieldnames or []):
            raise ValueError('Mapping requires slide_id,patient_id columns')
        patients={}
        for row in reader:
            sid,pid=row['slide_id'].strip(),row['patient_id'].strip()
            if not sid or not pid or sid in patients:
                raise ValueError('Empty/duplicate mapping identity')
            patients[sid]=pid
    if set(labels)!=set(patients):
        raise ValueError(f'Complete exact mapping required: {len(set(labels)-set(patients))} missing, {len(set(patients)-set(labels))} extra slides')
    if feature_format not in ('pt','h5'):
        raise ValueError('Feature format must be pt or h5')
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+'.tmp')
    try:
        with temp.open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['slide_id','patient_id','label','feature_path'])
            for sid in sorted(labels):
                writer.writerow([sid,patients[sid],labels[sid],str(root/feature_format/f'{sid}.{feature_format}')])
        slides,labels_by_patient=read_cohort(temp)
        temp.replace(output)
        return dict(slides=len(slides),patients=len(labels_by_patient))
    finally:
        temp.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--features',required=True); parser.add_argument('--mapping',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--format',choices=['pt','h5'],default='h5')
    a=parser.parse_args(); print(prepare(a.features,a.mapping,a.output,a.format))
