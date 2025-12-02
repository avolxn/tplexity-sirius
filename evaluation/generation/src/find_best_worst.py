#!/usr/bin/env python3
"""
Скрипт для поиска лучшего и худшего примеров из результатов оценки.
"""

import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any


def calculate_composite_score(row: pd.Series) -> float:
    """
    Вычисляет композитную оценку на основе всех метрик.
    
    Формула: (relevance + faithfulness + completeness + (1 - off_topic_rate)) / 4
    """
    relevance = row.get("relevance", 0.0)
    faithfulness = row.get("faithfulness", 0.0)
    completeness = row.get("completeness", 0.0)
    off_topic_rate = row.get("off_topic_rate", 0.0)
    
    # Преобразуем off_topic_rate: чем меньше, тем лучше (инвертируем)
    off_topic_score = 1.0 - off_topic_rate
    
    composite = (relevance + faithfulness + completeness + off_topic_score) / 4.0
    return composite


def load_generated_answers(generated_answers_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Загружает полные данные из generated_answers.jsonl, если файл существует.
    
    Returns:
        Словарь {query_id: example_dict}
    """
    if not generated_answers_path.exists():
        return {}
    
    examples_dict = {}
    with open(generated_answers_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                example = json.loads(line)
                examples_dict[idx] = example
    return examples_dict


def find_best_worst(
    results_path: str,
    output_dir: str,
    use_composite: bool = True,
    generated_answers_path: str = None
) -> None:
    """
    Находит лучший и худший примеры из результатов.
    
    Args:
        results_path: Путь к results.parquet
        output_dir: Директория для сохранения результатов
        use_composite: Использовать композитную оценку или отдельные метрики
        generated_answers_path: Путь к generated_answers.jsonl (опционально)
    """
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем результаты
    print(f"Загрузка результатов из {results_path}...")
    df = pd.read_parquet(results_path)
    print(f"Загружено {len(df)} примеров")
    
    # Загружаем полные данные, если доступны
    full_data = {}
    if generated_answers_path:
        gen_path = Path(generated_answers_path)
        full_data = load_generated_answers(gen_path)
        if full_data:
            print(f"Загружено {len(full_data)} полных примеров из {gen_path}")
        else:
            print(f"Файл {gen_path} не найден или пуст, используем только метрики")
    
    # Вычисляем композитную оценку
    if use_composite:
        df["composite_score"] = df.apply(calculate_composite_score, axis=1)
        score_col = "composite_score"
    else:
        # Используем среднее по основным метрикам
        df["mean_score"] = df[["relevance", "faithfulness", "completeness"]].mean(axis=1)
        score_col = "mean_score"
    
    # Находим лучший и худший примеры
    best_idx = df[score_col].idxmax()
    worst_idx = df[score_col].idxmin()
    
    best_row = df.loc[best_idx]
    worst_row = df.loc[worst_idx]
    
    # Получаем полные данные, если доступны
    best_query_id = int(best_row['query_id'])
    worst_query_id = int(worst_row['query_id'])
    best_full = full_data.get(best_query_id, {})
    worst_full = full_data.get(worst_query_id, {})
    
    print("\n" + "=" * 80)
    print("ЛУЧШИЙ ПРИМЕР")
    print("=" * 80)
    print(f"Query ID: {best_row['query_id']}")
    print(f"Composite Score: {best_row[score_col]:.4f}")
    print(f"Relevance: {best_row.get('relevance', 'N/A'):.4f}")
    print(f"Faithfulness: {best_row.get('faithfulness', 'N/A'):.4f}")
    print(f"Completeness: {best_row.get('completeness', 'N/A'):.4f}")
    print(f"Off-topic Rate: {best_row.get('off_topic_rate', 'N/A'):.4f}")
    print(f"Latency (ms): {best_row.get('latency_ms', 'N/A'):.2f}")
    print(f"\nQuestion: {best_row['question']}")
    if best_full.get('answer'):
        print(f"\nAnswer: {best_full['answer'][:500]}..." if len(best_full.get('answer', '')) > 500 else f"\nAnswer: {best_full.get('answer', 'N/A')}")
    if best_full.get('contexts'):
        print(f"\nContexts: {len(best_full['contexts'])} контекстов")
    
    print("\n" + "=" * 80)
    print("ХУДШИЙ ПРИМЕР")
    print("=" * 80)
    print(f"Query ID: {worst_row['query_id']}")
    print(f"Composite Score: {worst_row[score_col]:.4f}")
    print(f"Relevance: {worst_row.get('relevance', 'N/A'):.4f}")
    print(f"Faithfulness: {worst_row.get('faithfulness', 'N/A'):.4f}")
    print(f"Completeness: {worst_row.get('completeness', 'N/A'):.4f}")
    print(f"Off-topic Rate: {worst_row.get('off_topic_rate', 'N/A'):.4f}")
    print(f"Latency (ms): {worst_row.get('latency_ms', 'N/A'):.2f}")
    print(f"\nQuestion: {worst_row['question']}")
    if worst_full.get('answer'):
        print(f"\nAnswer: {worst_full['answer'][:500]}..." if len(worst_full.get('answer', '')) > 500 else f"\nAnswer: {worst_full.get('answer', 'N/A')}")
    if worst_full.get('contexts'):
        print(f"\nContexts: {len(worst_full['contexts'])} контекстов")
    
    # Сохраняем в JSON
    best_dict = best_row.to_dict()
    worst_dict = worst_row.to_dict()
    
    # Добавляем полные данные, если доступны
    if best_full:
        best_dict.update(best_full)
    if worst_full:
        worst_dict.update(worst_full)
    
    # Конвертируем numpy типы в Python типы для JSON
    def convert_types(obj):
        # Проверяем на pd.Timestamp
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        # Проверяем на NaT (Not a Time) - специальное значение pandas для отсутствующих дат
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        # Проверяем на numpy/pandas числовые типы
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    best_dict = {k: convert_types(v) for k, v in best_dict.items()}
    worst_dict = {k: convert_types(v) for k, v in worst_dict.items()}
    
    best_path = output_dir / "best_example.json"
    worst_path = output_dir / "worst_example.json"
    
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_dict, f, ensure_ascii=False, indent=2)
    print(f"\nЛучший пример сохранен в {best_path}")
    
    with open(worst_path, "w", encoding="utf-8") as f:
        json.dump(worst_dict, f, ensure_ascii=False, indent=2)
    print(f"Худший пример сохранен в {worst_path}")
    
    # Также сохраняем топ-5 и худших-5
    top5 = df.nlargest(5, score_col)
    bottom5 = df.nsmallest(5, score_col)
    
    top5_path = output_dir / "top5_examples.json"
    bottom5_path = output_dir / "bottom5_examples.json"
    
    top5_list = [convert_types(row.to_dict()) for _, row in top5.iterrows()]
    bottom5_list = [convert_types(row.to_dict()) for _, row in bottom5.iterrows()]
    
    with open(top5_path, "w", encoding="utf-8") as f:
        json.dump(top5_list, f, ensure_ascii=False, indent=2)
    print(f"Топ-5 примеров сохранены в {top5_path}")
    
    with open(bottom5_path, "w", encoding="utf-8") as f:
        json.dump(bottom5_list, f, ensure_ascii=False, indent=2)
    print(f"Худшие-5 примеров сохранены в {bottom5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Находит лучший и худший примеры из результатов оценки")
    parser.add_argument(
        "--results",
        type=str,
        default="eval/outputs/results.parquet",
        help="Путь к results.parquet (default: eval/outputs/results.parquet)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval/outputs",
        help="Директория для сохранения результатов (default: eval/outputs)"
    )
    parser.add_argument(
        "--no-composite",
        action="store_true",
        help="Использовать среднее по метрикам вместо композитной оценки"
    )
    parser.add_argument(
        "--generated-answers",
        type=str,
        default=None,
        help="Путь к generated_answers.jsonl (опционально, для полных данных)"
    )
    
    args = parser.parse_args()
    
    # Если не указан явно, пытаемся найти в той же директории
    generated_answers_path = args.generated_answers
    if not generated_answers_path:
        results_dir = Path(args.results).parent
        potential_path = results_dir / "generated_answers.jsonl"
        if potential_path.exists():
            generated_answers_path = str(potential_path)
    
    find_best_worst(
        results_path=args.results,
        output_dir=args.out,
        use_composite=not args.no_composite,
        generated_answers_path=generated_answers_path
    )

