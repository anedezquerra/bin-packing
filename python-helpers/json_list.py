import json
import argparse
import os
from collections import defaultdict, OrderedDict

# Define artifact directories
ARTIFACT_DIRS = [
    'artifacts/html',
    'artifacts/images',
    'artifacts/meshes',
    'artifacts/renders/gltf',
    'artifacts/renders/obj',
    'artifacts/renders/vtm'
]

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def remove_job_files(base_dir, job_id):
    """Remove job files from all artifact directories"""
    for dir_path in ARTIFACT_DIRS:
        # Get all possible files in the directory starting with job_id
        full_dir = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_dir):
            continue
            
        for filename in os.listdir(full_dir):
            if filename.startswith(job_id):
                file_path = os.path.join(full_dir, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {str(e)}")

def prune_null_entries(data, base_dir):
    to_delete = []
    for job_id, entry in data.items():
        if (entry.get('container_type') is None and 
            entry.get('structural_conservation_type') is None and 
            entry.get('solve_result_num') is None):
            to_delete.append(job_id)
    
    for job_id in to_delete:
        del data[job_id]
        remove_job_files(base_dir, job_id)
    
    return to_delete

def generate_stats(data):
    stats = {
        'total_entries': len(data),
        'type_combinations': defaultdict(int)
    }
    
    combinations_dict = defaultdict(int)
    for entry in data.values():
        key = (entry.get('container_type'), entry.get('structural_conservation_type'))
        combinations_dict[key] += 1
    
    # Sort the combinations by their keys (convert None to empty string for safe comparison)
    sorted_items = sorted(combinations_dict.items(),
                         key=lambda item: (
                             str(item[0][0]) if item[0][0] is not None else '',
                             str(item[0][1]) if item[0][1] is not None else ''
                         ))
    stats['type_combinations'] = OrderedDict(sorted_items)
        
    return stats

def validate_neos_results(data, base_dir):
    """Remove artifact files for jobs in neos_results but not in JSON"""
    neos_dir = os.path.join(base_dir, 'neos_results')
    if not os.path.exists(neos_dir):
        print(f"neos_results directory not found at {neos_dir}")
        return 0

    json_job_ids = set(data.keys())
    removed_count = 0
    
    # Process each file in neos_results
    for filename in os.listdir(neos_dir):
        job_id = os.path.splitext(filename)[0]
        
        # Skip if job exists in JSON
        if job_id in json_job_ids:
            continue
            
        # Remove associated files
        remove_job_files(base_dir, job_id)
        removed_count += 1
    
    return removed_count

def main():
    parser = argparse.ArgumentParser(description='Process job data JSON file')
    parser.add_argument('--input-file', required=True, help='JSON file to process')
    parser.add_argument('--base-dir', default='.', help='Base directory for artifact paths')
    parser.add_argument('--prune', action='store_true', help='Prune null entries and remove associated files')
    parser.add_argument('--stats', action='store_true', help='Generate statistics')
    parser.add_argument('--item', type=str, help='Show item by job_id')
    parser.add_argument('--del', dest='delete', type=str, help='Delete item by job_id and remove associated files')
    parser.add_argument('--list', type=int, help='List first/last N items')
    parser.add_argument('--val', action='store_true', help='Validate neos_results against JSON and clean orphaned files')
    
    args = parser.parse_args()
    data = load_data(args.input_file)
    base_dir = args.base_dir
    
    if args.prune:
        deleted_jobs = prune_null_entries(data, base_dir)
        save_data(args.input_file, data)
        print(f"Deleted {len(deleted_jobs)} entries with null values and their associated files")
    
    if args.stats:
        stats = generate_stats(data)
        print(f"Total entries: {stats['total_entries']}")
        print("Combination counts (container_type, structural_conservation_type):")
        for (c_type, s_type), count in stats['type_combinations'].items():
            print(f"  ({c_type}, {s_type}): {count}")
    
    if args.item:
        entry = data.get(args.item)
        if entry:
            print(json.dumps(entry, indent=2))
        else:
            print(f"Job ID {args.item} not found")
    
    if args.delete:
        if args.delete in data:
            del data[args.delete]
            save_data(args.input_file, data)
            remove_job_files(base_dir, args.delete)
            print(f"Deleted job ID {args.delete} and its associated files")
        else:
            print(f"Job ID {args.delete} not found")
    
    if args.list is not None:
        n = args.list
        items = list(data.items())
        if n < 0:
            result = dict(items[n:])
        else:
            result = dict(items[:n])
        print(json.dumps(result, indent=2))
    
    if args.val:
        removed_count = validate_neos_results(data, base_dir)
        print(f"Validated neos_results: Removed artifacts for {removed_count} orphaned jobs")

if __name__ == '__main__':
    main()
    
# Prune null entries (removes from JSON and artifacts)
#python script.py --input-file data.json --base-dir /path/to/project --prune

# Validate neos_results
#python script.py --input-file data.json --base-dir /path/to/project --val

# Delete specific job
#python script.py --input-file data.json --base-dir /path/to/project --del 15984287    