import time
import re

def original_snippet_logic(query, note_parts):
    results = []
    for note_part in note_parts:
        snippet = note_part[:200]
        if note_part:
            import re
            query_words = set(re.findall(r'\w+', query.lower()))
            if query_words:
                best_score = -1
                best_idx = 0
                max_len = 200
                step = 50
                for idx in range(0, max(1, len(note_part) - max_len + step), step):
                    window = note_part[idx:idx+max_len]
                    window_words = set(re.findall(r'\w+', window.lower()))
                    window_score = len(query_words & window_words)
                    if window_score > best_score:
                        best_score = window_score
                        best_idx = idx

                if best_score > 0:
                    start = best_idx
                    end = start + max_len
                    snippet = note_part[start:end].strip()
                    if start > 0:
                        snippet = "…" + snippet
                    if end < len(note_part):
                        snippet = snippet + "…"
        results.append(snippet)
    return results

def optimized_snippet_logic(query, note_parts):
    results = []
    import re
    word_pattern = re.compile(r'\w+')
    query_words = set(word_pattern.findall(query.lower()))

    for note_part in note_parts:
        snippet = note_part[:200]
        if note_part and query_words:
            best_score = -1
            best_idx = 0
            max_len = 200
            step = 50
            for idx in range(0, max(1, len(note_part) - max_len + step), step):
                window = note_part[idx:idx+max_len]
                window_words = set(word_pattern.findall(window.lower()))
                window_score = len(query_words & window_words)
                if window_score > best_score:
                    best_score = window_score
                    best_idx = idx

            if best_score > 0:
                start = best_idx
                end = start + max_len
                snippet = note_part[start:end].strip()
                if start > 0:
                    snippet = "…" + snippet
                if end < len(note_part):
                    snippet = snippet + "…"
        results.append(snippet)
    return results

query = "performance optimization with regex inside loops"
note_part = "The performance optimization inside the loop can be achieved by pulling out the regex. " * 100
note_parts = [note_part] * 100

start = time.perf_counter()
original_snippet_logic(query, note_parts)
end = time.perf_counter()
print(f"Original: {end - start:.4f} seconds")

start = time.perf_counter()
optimized_snippet_logic(query, note_parts)
end = time.perf_counter()
print(f"Optimized: {end - start:.4f} seconds")
