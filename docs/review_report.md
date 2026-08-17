# 🔍 Forensic Review & Infrastructure Audit Report — AIVC Evaluation Suite

> **Document Status**: RUTHLESS AUDIT & FORENSIC GAP ANALYSIS  
> **Target Conferences**: IEEE/ACM MSR 2027 / AAMAS 2027 / EASE 2027  
> **Audited Modules**: `paper/`, `eval/benchmarks/swebench_cl_runner.py`, `eval/benchmarks/devbench_runner.py`, `eval/benchmarks/agentic_rag_runner.py`, `eval/metrics/`, `eval/plots/`, `eval/checkpoints/`, `eval/config/`  
> **Verdict**: ❌ **REJETÉ POUR BENCHMARK DE PRODUCTION (N=273)** — 8 Failles Bloquantes Identifiées.

---

## Executive Summary

Avant de lancer le benchmark de production complet ($N=273$ instances, estimé à 9 heures de calcul et plusieurs centaines de dollars d'API), cet audit médico-légal et adversarial a analysé l'infrastructure de collecte de données et les artefacts de sortie d'AIVC par rapport aux affirmations scientifiques du papier.

**Constat majeur** : En l'état actuel, **plus de 60% des figures, tables et métriques promises dans le papier sont impossibles à générer** à partir des sorties de données actuelles, et **une interruption en cours de run corrompt irrémédiablement l'état d'apprentissage continu** lors de la reprise.

---

## 📊 Matrice d'Audit des Figures, Tables & Déclarations Scientifiques

| Élément Scientifique Promis dans le Papier | Source de Données Actuelle | Statut d'Exécutabilité | Faille / Variable Manquante |
| :--- | :--- | :---: | :--- |
| **Figure 1 : Courbe de décroissance des appels d'outils ($\mathcal{O}(N) \to \mathcal{O}(1)$)** | `eval/plots/swebench_cl_curves.csv` | 🔴 **IMPOSSIBLE** | `tool_calls_count`, `exploration_tool_calls` et `tool_call_decay_ratio` sont **totalement absents** de `swebench_cl_curves.csv` et `devbench_curves.csv`. |
| **Table 1 : Comparatif 4-voies (AIVC vs Stateless vs FAISS RAG vs Flat Memory)** | `eval/metrics/summary_metrics.json` | 🔴 **IMPOSSIBLE** | `swebench_cl_runner.py` et `devbench_runner.py` n'ont **aucun argument `--arm`**. Seul AIVC est exécuté. Zéro baseline générée. |
| **Figure 2 : Évolution des coûts de tokens par tour (Prompt vs Completion)** | `eval/checkpoints/*.jsonl` | 🔴 **IMPOSSIBLE** | La ventilation tour par tour (`trajectory_steps`) est calculée en mémoire vive mais **supprimée/ignorée** lors de l'écriture du checkpoint JSONL. |
| **Table 2 : Matrice de Continual Learning ($a_{i,j}$, $CL\text{-}P$, $CL\text{-}S$, $F$, $CL\text{-}F_\beta$)** | `eval/metrics/swebench_cl_metrics.json` | 🔴 **IMPOSSIBLE** | Aucune implémentation de la matrice $a_{i,j}$ ni de la mesure de régression rétrospective ($F$). Seul le Pass@1 cumulé est calculé. |
| **Figure 3 : Transfert Inter-Phases DevBench (Design $\to$ Code $\to$ Test)** | `eval/metrics/devbench_metrics.json` | 🔴 **IMPOSSIBLE** | Le runner DevBench n'enregistre aucune traçabilité de provenance des mémoires entre phases (intra-phase uniquement). |
| **Figure 4 : Dérive Contextuelle & Poisoning Resilience (RAG vs AIVC)** | `eval/plots/agentic_rag_curves.csv` | 🟡 **PARTIEL** | Les courbes Naive et AIVC s'écrasent mutuellement dans le même fichier CSV au lieu de générer un fichier comparatif multi-colonnes. |
| **Table 3 : Efficacité des requêtes IR (P@1, P@3, R@1, R@3, MRR)** | `eval/metrics/agentic_rag_metrics.json` | 🟢 **FONCTIONNEL** | IR metrics (P@k, R@k, MRR) correctement implémentées et exportées dans `agentic_rag_runner.py`. |

---

## 🚨 Classification des Problèmes Identifiés

---

### 🔴 1. PROBLÈMES BLOQUANTS (Arrêt Immédiat de la Production)

#### 🔴 Issue #1 : Désynchronisation de la Mémoire AIVC lors de la Reprise sur Checkpoint (Crash-Recovery Flaw)
* **Symptôme** : Si un benchmark de 273 instances s'interrompt à l'instance $k=150$ et est relancé :
  - `CheckpointManager` détecte les 150 premières instances et les saute (`[SKIP]`).
  - Cependant, `runner.aivc_env = AIVCEnvironment()` démarre avec une **mémoire vide** (`self.memories = {}`).
  - L'instance 151 s'exécute donc comme une instance "cold start" (sans aucune des 150 mémoires passées).
* **Preuve dans le Code** (`swebench_cl_runner.py:905-938`) :
  ```python
  # Initialisation d'une mémoire vide à chaque démarrage de script
  runner = SWEBenchCLRunner(...) # self.aivc_env = AIVCEnvironment()
  for idx, inst in enumerate(instances, 1):
      if ckpt_mgr.is_processed(inst_id):
          continue # Saute l'instance MAIS ne réhydrate PAS la mémoire !
      episode_rec = runner.run_episode(...) # Exécuté avec 0 mémoire passée !
  ```
* **Impact** : Invalidation scientifique totale du caractère "Continual Learning" en cas de reprise après crash.

---

#### 🔴 Issue #2 : Perte Sèche de la Télémétrie Tour par Tour (Prompt vs Completion Token Breakdown)
* **Symptôme** : L'agent accumule les tokens tour par tour dans `trajectory_steps = []` pendant l'interaction ReAct. Mais lors de la création de `episode_record`, `trajectory_steps` est **omis** du dictionnaire sauvegardé dans le JSONL.
* **Preuve dans le Code** (`swebench_cl_runner.py:648-697`, `devbench_runner.py:606-653`, `agentic_rag_runner.py:1120-1187`) :
  ```python
  # trajectory_steps contient les détails tour par tour :
  trajectory_steps.append({
      "turn": turn,
      "tool_calls": turn_tool_names,
      "prompt_tokens": p_tok,
      "completion_tokens": c_tok,
      "recalled_memories": turn_recalled,
      "used_memories": turn_used,
  })
  # MAIS episode_record ne conserve QUE les sommes globales :
  episode_record = {
      ...
      "tokens": {
          "prompt_tokens": total_prompt_tokens,
          "completion_tokens": total_completion_tokens,
          "total_tokens": total_prompt_tokens + total_completion_tokens,
          "cost_usd": round(total_instance_cost, 6),
      },
      # trajectory_steps N'EST PAS SAUVEGARDÉ !
  }
  ```
* **Impact** : Impossible de tracer les courbes d'expansion du prompt et de charge cognitive au fil des tours d'une tâche.

---

#### 🔴 Issue #3 : Absence Totale des Métriques de Décroissance d'Appels d'Outils dans les CSVs SWE-bench-CL et DevBench
* **Symptôme** : `swebench_cl_curves.csv` et `devbench_curves.csv` n'exportent pas les colonnes `tool_calls_count`, `exploration_tool_calls`, ou `tool_call_decay_ratio`.
* **Preuve dans le Code** (`swebench_cl_runner.py:767-780`) :
  ```python
  fieldnames = [
      "episode_index",
      "instance_id",
      "repo",
      "timestamp",
      "resolved",
      "cumulative_resolved",
      "resolve_rate",
      "cumulative_cost_usd",
      "cumulative_eor",
      "cumulative_mui",
      "cumulative_ccsr",
  ] # Aucune colonne sur le volume d'appels d'outils !
  ```
* **Impact** : Impossible de générer la figure maîtresse démontrant la transition $\mathcal{O}(N) \to \mathcal{O}(1)$.

---

#### 🔴 Issue #4 : Absence d'Options de Baselines / Ablations dans SWE-bench-CL et DevBench
* **Symptôme** : `swebench_cl_runner.py` et `devbench_runner.py` ne supportent aucun flag `--arm` (contrairement à `agentic_rag_runner.py`). Il est impossible d'exécuter la baseline sans mémoire (*stateless zero-memory*), l'ablation RAG vectoriel plat (*FAISS RAG*), ou l'ablation journal plat non indexé (*flat unstructured*).
* **Impact** : Aucune donnée comparative pour bâtir les tables du papier.

---

#### 🔴 Issue #5 : Écrasement des Fichiers de Courbes CSV lors de l'Évaluation Multi-Bras
* **Symptôme** : Dans `agentic_rag_runner.py`, exécuter `--arm naive` puis `--arm aivc` écrase `eval/plots/agentic_rag_curves.csv` au lieu de créer des fichiers distincts (`agentic_rag_curves_aivc.csv` vs `agentic_rag_curves_naive.csv`) ou de fusionner les colonnes.
* **Preuve dans le Code** (`agentic_rag_runner.py:1586-1598`) :
  ```python
  export_agentic_rag_curves(
      records=all_records,
      curves_path=general_curves_path, # Écrase sans distinction de l'arm !
  )
  ```
* **Impact** : Perte des courbes de baseline à chaque nouveau run.

---

#### 🔴 Issue #6 : Pollution Mémorielle Inter-Dépôts dans DevBench et SWE-bench-CL
* **Symptôme** : `DevBenchAIVCEnvironment` et `AIVCEnvironment` stockent toutes les mémoires dans un unique dictionnaire global non partitionné par dépôt. Lors de l'exécution de 8 dépôts distincts (Django, SymPy, Matplotlib), un rappel sur Django peut retourner des mémoires de SymPy.
* **Impact** : Bruit contextuel artificiel non représentatif des conditions réelles.

---

#### 🔴 Issue #7 : Faux Succès de Résolution (Pass@1 basé sur `submit_patch` au lieu d'une validation Pytest Docker)
* **Symptôme** : Dans `swebench_cl_runner.py`, une tâche est marquée `resolved: True` dès que le modèle appelle l'outil `submit_patch`, sans aucune validation par exécution de tests unitaires Docker (`FAIL_TO_PASS` / `PASS_TO_PASS`).
* **Preuve dans le Code** (`swebench_cl_runner.py:626-628`) :
  ```python
  if fn_name == "submit_patch":
      resolved = True
      submitted_patch = fn_args.get("patch", "")
  ```
* **Impact** : Le taux de résolution rapporté mesure le taux de soumission de patch et non le taux réel de succès logiciel (Pass@1).

---

#### 🔴 Issue #8 : Marquage Définitif d'Échec sur Panne Réseau / Rate Limit
* **Symptôme** : Si l'API renvoie des erreurs répétées (5 timeouts / 429), le runner interrompt la tâche et l'enregistre dans le checkpoint comme `unresolved`. L'instance ne sera plus jamais réessayée lors des runs ultérieurs.
* **Impact** : Dégradation artificielle des métriques scientifiques suite à des aléas réseau.

---

### 🟡 2. PROBLÈMES MINEURS

1. **🟡 Tarification Fallback Hardcodée dans DVC Exporter** : `dvc_exporter.py` utilise par défaut `$0.03 / $0.13` par 1M tokens (Qwen) pour imputer les coûts si les données de coûts sont manquantes, faussant les coûts pour GPT-5.6 ($1.50/$6.00) ou Claude Sonnet 5 ($3.00/$15.00).
2. **🟡 Absence d'Horodatage Sub-Épisode** : Seul l'horodatage de fin d'épisode est conservé, empêchant l'analyse fine de la latence inter-outils.
3. **🟡 Résidus de Code InterCode Déprécié** : Le runner InterCode est marqué déprécié mais continue d'être référencé dans certains fallbacks de `dvc_exporter.py`.

---

### 🟠 3. AMÉLIORATIONS RECOMMANDÉES

1. **🟠 Replay de Réhydratation Automatique du Checkpoint** : Lors de la reprise après crash, le `CheckpointManager` doit réinjecter les notes des instances passées dans `AIVCEnvironment` avant de lancer la suite.
2. **🟠 Partitionnement des Espaces de Noms Mémoriels (`repo_id`)** : Indexer les mémoires avec un tag de dépôt pour isoler strictement les contextes entre projets.
3. **🟠 Pipeline DVC Multi-Bras et Multi-Modèles** : Structurer `dvc.yaml` avec une matrice d'exécution explicite couvrant les 4 baselines et les modèles cibles.

---

## 📋 Checklist Pré-Benchmark de Production ($N=273$)

- [ ] **1. Télémétrie Complète** : Inclure `trajectory_steps` dans les enregistrements JSONL de tous les runners.
- [ ] **2. Sécurisation de la Reprise (Crash Recovery)** : Implémenter la réhydratation de la mémoire AIVC au démarrage à partir des checkpoints existants.
- [ ] **3. Métriques Décroissance d'Outils** : Ajouter les colonnes `tool_calls_count`, `exploration_tool_calls`, et `tool_call_decay_ratio` dans tous les fichiers CSV de courbes.
- [ ] **4. Intégration des Bras de Baseline (`--arm`)** : Ajouter le support des modes `stateless`, `faiss_rag`, et `flat_memory` dans `swebench_cl_runner.py` et `devbench_runner.py`.
- [ ] **5. Isolation des Sorties Multi-Bras** : Générer des fichiers de métriques et de courbes nommés par bras et par modèle (ex: `swebench_cl_curves_{model}_{arm}.csv`).
- [ ] **6. Isolation Mémorielle par Dépôt** : Filtrer le rappel mémoriel par `repo_id`.
- [ ] **7. Gestion des Erreurs Fatales d'Inférence** : Ne pas enregistrer dans le checkpoint les instances interrompues par une panne API.
