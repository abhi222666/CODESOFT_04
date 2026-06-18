import math

USER_RATINGS = {
    "Alice": {"Inception": 5, "Interstellar": 4, "The Matrix": 5, "Avengers": 3},
    "Bob":   {"Inception": 4, "Interstellar": 5, "The Matrix": 4, "Gravity": 5, "Doctor Strange": 4},
    "Carol": {"Titanic": 5, "The Notebook": 5, "Dunkirk": 4, "Interstellar": 2},
    "Dave":  {"Inception": 5, "The Matrix": 4, "Avengers": 5, "Iron Man": 5, "Doctor Strange": 3},
    "Eve":   {"Titanic": 5, "The Notebook": 4, "Inception": 2, "Avengers": 1},
}

MOVIE_FEATURES = {
    "Inception":      {"sci-fi", "thriller", "mind-bending", "action"},
    "Interstellar":   {"sci-fi", "space", "drama", "mind-bending"},
    "The Matrix":     {"sci-fi", "action", "cyberpunk", "mind-bending"},
    "Titanic":        {"romance", "drama", "historical", "disaster"},
    "Avengers":       {"action", "superhero", "sci-fi", "adventure"},
    "Gravity":        {"sci-fi", "space", "thriller", "drama"},
    "The Notebook":   {"romance", "drama", "historical"},
    "Iron Man":       {"action", "superhero", "sci-fi", "adventure"},
    "Dunkirk":        {"historical", "war", "thriller", "drama"},
    "Doctor Strange": {"sci-fi", "superhero", "action", "mind-bending"},
}

def cosine_similarity(ratings_a: dict, ratings_b: dict) -> float:
    common = set(ratings_a) & set(ratings_b)
    if not common:
        return 0.0
    dot   = sum(ratings_a[m] * ratings_b[m] for m in common)
    mag_a = math.sqrt(sum(ratings_a[m] ** 2 for m in common))
    mag_b = math.sqrt(sum(ratings_b[m] ** 2 for m in common))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def jaccard_similarity(features_a: set, features_b: set) -> float:
    if not features_a or not features_b:
        return 0.0
    return len(features_a & features_b) / len(features_a | features_b)


def collaborative_recommend(target_user: str, top_n: int = 3):
    if target_user not in USER_RATINGS:
        print(f"  User '{target_user}' not found.")
        return []

    target_ratings = USER_RATINGS[target_user]
    seen_movies    = set(target_ratings.keys())

    similarities = {}
    for user, ratings in USER_RATINGS.items():
        if user == target_user:
            continue
        sim = cosine_similarity(target_ratings, ratings)
        if sim > 0:
            similarities[user] = sim

    if not similarities:
        print("  No similar users found.")
        return []

    scores   = {}
    sim_sums = {}
    for user, sim in similarities.items():
        for movie, rating in USER_RATINGS[user].items():
            if movie not in seen_movies:
                scores.setdefault(movie, 0)
                sim_sums.setdefault(movie, 0)
                scores[movie]   += sim * rating
                sim_sums[movie] += sim

    if not scores:
        print("  No unseen movies to recommend (user has rated everything!).")
        return []

    weighted = {m: scores[m] / sim_sums[m] for m in scores}
    ranked   = sorted(weighted, key=weighted.get, reverse=True)[:top_n]
    return [(m, round(weighted[m], 2)) for m in ranked]


def content_recommend(liked_movie: str, top_n: int = 3):
    if liked_movie not in MOVIE_FEATURES:
        print(f"  Movie '{liked_movie}' not found in database.")
        return []

    base_features = MOVIE_FEATURES[liked_movie]
    similarities  = {
        movie: jaccard_similarity(base_features, features)
        for movie, features in MOVIE_FEATURES.items()
        if movie != liked_movie
    }

    ranked = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
    return [(m, round(similarities[m], 2)) for m in ranked]


def show_user_ratings():
    all_movies = sorted(MOVIE_FEATURES.keys())
    col_w = 14
    print(f"\n{'User':<10}", end="")
    for m in all_movies:
        print(f"{m[:col_w-2]:<{col_w}}", end="")
    print()
    print("-" * (10 + col_w * len(all_movies)))
    for user, ratings in USER_RATINGS.items():
        print(f"{user:<10}", end="")
        for m in all_movies:
            val = str(ratings.get(m, "-"))
            print(f"{val:<{col_w}}", end="")
        print()

def show_results(title, results):
    print(f"\n🎬 {title}:")
    if not results:
        print("  No recommendations found.")
        return
    for i, (movie, score) in enumerate(results, 1):
        print(f"  {i}. {movie:<22} (score: {score})")


def menu():
    print("\n")
    print("      Movie Recommendation System")

    while True:
        print("\nOptions:")
        print("  1. Collaborative filtering  (recommend for a user)")
        print("  2. Content-based filtering  (find similar movies)")
        print("  3. Show all user ratings")
        print("  4. Quit")

        choice = input("\nChoose (1-4): ").strip()

        if choice == "1":
            print(f"\n  Available users: {', '.join(USER_RATINGS.keys())}")
            user = input("  Enter username: ").strip()
            results = collaborative_recommend(user, top_n=3)
            show_results(f"Collaborative recommendations for {user}", results)

        elif choice == "2":
            print(f"\n  Available movies: {', '.join(MOVIE_FEATURES.keys())}")
            movie = input("  Enter a movie you liked: ").strip()
            results = content_recommend(movie, top_n=3)
            show_results(f"Movies similar to '{movie}'", results)

        elif choice == "3":
            show_user_ratings()

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("  Invalid choice. Enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    menu()
