import os
import pathlib
import fnmatch
from datetime import datetime

def read_gitignore_patterns(gitignore_path):
    """Legge e parserizza i pattern dal file .gitignore"""
    patterns = []
    if not os.path.exists(gitignore_path):
        return patterns
    
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Salta linee vuote e commenti
            if not line or line.startswith('#'):
                continue
            patterns.append(line)
    return patterns

def get_extra_ignore_patterns():
    """Restituisce i pattern aggiuntivi da ignorare (git, uv.lock, etc.)"""
    return [
        '.git/',           # Directory git
        '.git/*',          # Tutto dentro .git
        'uv.lock',         # File lock di uv
        '.python-version', # File versione Python
        '.gitignore'       # Il file gitignore stesso
    ]

def is_ignored(path, gitignore_patterns, root_path):
    """Controlla se un path corrisponde ai pattern del .gitignore"""
    # Converti il path in relativo rispetto alla root
    try:
        relative_path = os.path.relpath(path, root_path)
    except ValueError:
        return False
    
    # Pattern aggiuntivi da ignorare sempre
    extra_patterns = get_extra_ignore_patterns()
    all_patterns = gitignore_patterns + extra_patterns
    
    # Per directory, aggiungi '/' per matchare pattern di directory
    if os.path.isdir(path):
        check_path = relative_path + '/'
    else:
        check_path = relative_path
    
    for pattern in all_patterns:
        # Gestisci pattern che iniziano con /
        if pattern.startswith('/'):
            pattern = pattern[1:]
        
        # Gestisci pattern che finiscono con / (solo directory)
        if pattern.endswith('/'):
            if os.path.isdir(path) and fnmatch.fnmatch(relative_path, pattern[:-1]):
                return True
            continue
        
        # Match standard con fnmatch
        if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(check_path, pattern):
            return True
        
        # Match per file in directory ignorata
        if any(fnmatch.fnmatch(part, pattern) for part in relative_path.split(os.sep)):
            return True
    
    return False

def get_file_content(file_path):
    """Legge il contenuto di un file, gestendo file binari e errori di encoding"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        return "[BINARY FILE - CONTENT OMITTED]"
    except Exception as e:
        return f"[ERROR READING FILE: {str(e)}]"

def build_directory_tree(root_path, gitignore_patterns, prefix="", is_last=True, output_lines=None):
    """Costruisce la struttura ad albero delle directory"""
    if output_lines is None:
        output_lines = []
    
    # Determina il nome da mostrare (usa il nome della directory per la root)
    if prefix == "":
        display_name = os.path.basename(root_path) if root_path != "." else "PROJECT_ROOT"
    else:
        display_name = os.path.basename(root_path)
    
    # Aggiungi la linea corrente
    connector = "└── " if is_last else "├── "
    output_lines.append(f"{prefix}{connector}{display_name}/")
    
    # Prepara il prefisso per i figli
    new_prefix = prefix + ("    " if is_last else "│   ")
    
    try:
        # Lista tutti gli elementi, escludendo quelli ignorati
        items = []
        for item in sorted(os.listdir(root_path)):
            item_path = os.path.join(root_path, item)
            if not is_ignored(item_path, gitignore_patterns, os.path.dirname(root_path)):
                items.append(item)
        
        # Processa ogni elemento
        for i, item in enumerate(items):
            item_path = os.path.join(root_path, item)
            is_last_item = i == len(items) - 1
            
            if os.path.isdir(item_path):
                build_directory_tree(item_path, gitignore_patterns, new_prefix, is_last_item, output_lines)
            else:
                connector = "└── " if is_last_item else "├── "
                output_lines.append(f"{new_prefix}{connector}{item}")
    
    except PermissionError:
        output_lines.append(f"{new_prefix}└── [Permission Denied]")
    
    return output_lines

def generate_project_log(output_file="scripts/output/project_log.txt"):
    """Genera il log completo del progetto"""
    root_path = os.getcwd()
    gitignore_path = os.path.join(root_path, ".gitignore")
    
    # Leggi i pattern del .gitignore
    gitignore_patterns = read_gitignore_patterns(gitignore_path)
    
    # Crea il log
    log_content = []
    log_content.append("=" * 80)
    log_content.append(f"PROJECT STRUCTURE LOG")
    log_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append(f"Project Root: {root_path}")
    log_content.append("=" * 80)
    log_content.append("")
    
    # Aggiungi la struttura ad albero
    log_content.append("DIRECTORY STRUCTURE:")
    log_content.append("")
    tree_lines = build_directory_tree(root_path, gitignore_patterns)
    log_content.extend(tree_lines)
    log_content.append("")
    
    # Aggiungi il contenuto dei file
    log_content.append("=" * 80)
    log_content.append("FILE CONTENTS:")
    log_content.append("=" * 80)
    log_content.append("")
    
    # Percorsi di file già processati (per evitare duplicati)
    processed_files = set()
    
    def process_directory(dir_path):
        """Processa ricorsivamente una directory"""
        try:
            for item in sorted(os.listdir(dir_path)):
                item_path = os.path.join(dir_path, item)
                
                # Salta se ignorato o già processato
                if (is_ignored(item_path, gitignore_patterns, root_path) or 
                    item_path in processed_files):
                    continue
                
                processed_files.add(item_path)
                
                if os.path.isfile(item_path):
                    # Aggiungi il contenuto del file
                    log_content.append(f"FILE: {os.path.relpath(item_path, root_path)}")
                    log_content.append("-" * 60)
                    
                    content = get_file_content(item_path)
                    if content.strip():
                        log_content.append(content)
                    else:
                        log_content.append("[EMPTY FILE]")
                    
                    log_content.append("\n" + "=" * 80 + "\n")
                
                elif os.path.isdir(item_path):
                    process_directory(item_path)
        
        except PermissionError:
            log_content.append(f"[Permission denied accessing: {dir_path}]")
    
    process_directory(root_path)
    
    # Scrivi il log su file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))
    
    print(f"Project log generated: {output_file}")
    print(f"Total files processed: {len(processed_files)}")
    print(f"Ignored patterns: {get_extra_ignore_patterns()}")

if __name__ == "__main__":
    generate_project_log()