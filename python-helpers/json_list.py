import json
import argparse
from collections import defaultdict, OrderedDict

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def prune_null_entries(data):
    to_delete = []
    for job_id, entry in data.items():
        if (entry.get('container_type') is None and 
            entry.get('structural_conservation_type') is None and 
            entry.get('solve_result_num') is None):
            to_delete.append(job_id)
    
    for job_id in to_delete:
        del data[job_id]
    
    return len(to_delete)

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

def main():
    parser = argparse.ArgumentParser(description='Process job data JSON file')
    parser.add_argument('filename', help='JSON file to process')
    parser.add_argument('--prune', action='store_true', help='Prune null entries')
    parser.add_argument('--stats', action='store_true', help='Generate statistics')
    parser.add_argument('--item', type=str, help='Show item by job_id')
    parser.add_argument('--del', dest='delete', type=str, help='Delete item by job_id')
    parser.add_argument('--list', type=int, help='List first/last N items')
    
    args = parser.parse_args()
    data = load_data(args.filename)
    
    if args.prune:
        deleted_count = prune_null_entries(data)
        save_data(args.filename, data)
        print(f"Deleted {deleted_count} entries with null values")
    
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
            save_data(args.filename, data)
            print(f"Deleted job ID {args.delete}")
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

if __name__ == '__main__':
    main()
    
# # Prune null entries
# python script.py data.json --prune

# # Show statistics
# python script.py data.json --stats

# # Show specific item
# python script.py data.json --item 15984287

# # Delete item
# python script.py data.json --del 15984287

# # List first 5 items
# python script.py data.json --list 5

# # List last 3 items
# python script.py data.json --list -3