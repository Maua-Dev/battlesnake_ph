# Bem-vindo ao
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \   __\   __\  | _/ __ \ /  ___//    \__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\_____>
#
# ESTE É O ARQUIVO QUE VOCÊ VAI EDITAR. Todo o resto do projeto existe
# só para levar o estado do jogo até as quatro funções (info, start, end, get_move).
#
# Estratégia: para cada movimento legal calculamos uma pontuação somando componentes
# (espaço, território Voronoi, comida via Dijkstra, confronto de cabeças, hazards...)
# e escolhemos o maior. Se algo falhar, caímos num fallback seguro.
# Documentação: https://docs.battlesnake.com

import heapq
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field

from .models import GameState, MoveResponse

logger = logging.getLogger(__name__)
# O runtime Python da Lambda deixa o logger raiz em WARNING: sem esta linha
# as jogadas nao aparecem no CloudWatch.
logger.setLevel(logging.INFO)

Pt = tuple[int, int]

# --------------------------------------------------------------------------- #
# CONFIGURAÇÃO (tudo que você vai querer ajustar fica aqui)
# --------------------------------------------------------------------------- #

MOVES: dict[str, Pt] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
MOVE_ORDER = ["up", "down", "left", "right"]
INF = 10**6
MAX_HEALTH = 100

# Pesos da pontuação. Escala aproximada: "mortal" ~ 700-900, "importante" ~ 100-300,
# "desempate" ~ 10-40. Não são valores matematicamente ótimos: ajuste com partidas de teste.
WEIGHTS: dict[str, float] = {
    # --- sobrevivência ---
    "h2h_loss": 900.0,    # casa que uma cobra MAIOR pode alcançar: perderíamos o head-to-head
    "h2h_tie": 700.0,     # idem com cobra de MESMO tamanho: as duas morrem
    "dead_end": 700.0,    # penalidade máxima quando a região acessível é menor que o corpo
    "space": 200.0,       # espaço acessível (flood fill), saturando em um "conforto"
    "mobility": 20.0,     # por saída livre logo após o movimento (evita corredores de 1 casa)
    "tail_reach": 40.0,   # bônus por conseguir alcançar a própria cauda (rota de fuga)
    # --- território / posição ---
    "territory": 250.0,   # fatia do tabuleiro que chegamos ANTES dos adversários (Voronoi)
    "center": 25.0,       # leve preferência pelo centro (some quando precisa de comida)
    "edge": 12.0,         # leve penalidade por borda (menos opções de fuga)
    # --- comida / saúde ---
    "food": 300.0,        # valor da melhor comida alcançável (modulado por urgência)
    "starvation": 300.0,  # não chegaremos a nenhuma comida antes de morrer de fome
    "low_health": 60.0,   # penalidade crescente conforme a vida cai de LOW_HEALTH
    # --- hazards ---
    "hazard": 60.0,       # custo de entrar em hazard (cresce quando a vida está baixa)
    # --- adversários ---
    "kill": 250.0,        # chance de eliminar cobra MENOR via head-to-head
    "threat": 60.0,       # proximidade da cabeça de cobra maior/igual
    "hunt": 80.0,         # aproximação de cabeça de cobra menor (quando saudável)
    # --- tática de cauda ---
    "tail_follow": 120.0, # seguir a própria cauda quando o espaço está apertado
}

# Limiares e fatores (não são "pesos", mas também são ajustáveis).
TUNING: dict[str, float] = {
    "health_critical": 25,      # abaixo disso, urgência de comida = 1.0
    "health_comfort": 65,       # acima disso, urgência = 0.0
    "hazard_urgency_shift": 15, # em mapas com hazard a vida se esgota mais rápido
    "food_base": 0.25,          # interesse mínimo em comida (crescer) mesmo com vida cheia
    "food_base_ahead": 0.10,    # idem quando já somos bem maiores que todos
    "scarce_food": 2,           # com <= N comidas no mapa, comida vale um pouco mais
    "contested": 0.2,           # fator se um adversário chega antes (ou empata e é >= a nós)
    "hazard_food": 0.4,         # fator para comida dentro de hazard (se não urgente)
    "deadend_food": 0.4,        # fator para comida em beco (<= 1 saída)
    "low_health": 20,           # abaixo disso começamos a penalizar vida baixa
    "threat_range": 3,          # distância em que cabeças maiores nos assustam
    "hunt_range": 4,            # distância em que caçamos cabeças menores
    "space_comfort_min": 20,    # espaço "confortável" mínimo (cresce com o tamanho)
    "tail_min_len": 8,          # só seguimos a cauda ativamente com corpo desse tamanho
    "opening_turns": 12,        # turnos considerados "abertura"
    "time_budget": 0.4,         # fração do timeout que podemos gastar calculando
}

# Multiplicadores por fase da partida (só altera o que for listado).
PHASE_MODS: dict[str, dict[str, float]] = {
    "opening": {"food": 1.3, "hunt": 0.3, "kill": 0.7},
    "midgame": {},
    "endgame": {"hunt": 2.0, "territory": 1.3, "food": 0.8},  # 1v1: pressionar o adversário
}


# --------------------------------------------------------------------------- #
# INFO / START / END
# --------------------------------------------------------------------------- #

def info() -> dict:
    return {
        "apiversion": "1",
        "author": "",          # TODO: coloque aqui o SEU usuário do Battlesnake
        "color": "#8B0000",
        "head": "tiger-king",
        "tail": "hook",
        "version": "1.2.0",
    }


def start(state: GameState) -> None:
    logger.info("JOGO COMEÇOU (partida %s)", state.game.id)


def end(state: GameState) -> None:
    logger.info("FIM DE JOGO após %d turnos", state.turn)


# --------------------------------------------------------------------------- #
# MODELOS INTERNOS
# --------------------------------------------------------------------------- #

@dataclass
class Enemy:
    id: str
    head: Pt
    body: list
    length: int
    health: int
    next_cells: list = field(default_factory=list)  # casas que a cabeça pode ocupar no próximo turno


@dataclass
class Context:
    width: int
    height: int
    turn: int
    wrapped: bool
    constrictor: bool
    hazard_damage: int
    map_name: str
    my_head: Pt
    my_tail: Pt
    my_body: list
    my_len: int
    my_health: int
    enemies: list
    food: set
    hazards: set
    free_at: dict          # casa -> turno (após o movimento) em que deixa de estar ocupada
    deadline: float
    phase: str = "midgame"
    urgency: float = 0.0
    w: dict = field(default_factory=dict)              # pesos já ajustados pela fase
    enemy_arrival: dict = field(default_factory=dict)  # casa -> (turno de chegada, tamanho)
    max_enemy_len: int = 0


@dataclass
class FloodResult:
    arrival: dict          # casa -> turno em que chegamos nela
    count: int
    tail_reachable: bool


@dataclass
class MoveEvaluation:
    move: str
    pos: Pt
    score: float
    parts: dict


# --------------------------------------------------------------------------- #
# LEITURA SEGURA DO STATE E CONSTRUÇÃO DO CONTEXTO
# --------------------------------------------------------------------------- #

def _dig(obj, *names, default=None):
    """Acessa obj.a.b.c (ou dict['a']['b']) sem estourar erro; devolve default se faltar."""
    for name in names:
        if obj is None:
            return default
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return default if obj is None else obj


def _pt(coord) -> Pt:
    return (coord["x"], coord["y"]) if isinstance(coord, dict) else (coord.x, coord.y)


def build_context(state: GameState, started: float) -> Context:
    board = state.board
    you = state.you
    my_id = _dig(you, "id")

    my_body = [_pt(c) for c in you.body]
    bodies = [my_body]
    enemies: list[Enemy] = []
    for snake in _dig(board, "snakes", default=[]):
        if _dig(snake, "id") == my_id:
            continue
        body = [_pt(c) for c in snake.body]
        if not body:
            continue
        bodies.append(body)
        enemies.append(Enemy(
            id=_dig(snake, "id", default=""),
            head=body[0],
            body=body,
            length=len(body),
            health=_dig(snake, "health", default=MAX_HEALTH),
        ))

    ruleset = str(_dig(state, "game", "ruleset", "name", default="standard")).lower()
    settings = _dig(state, "game", "ruleset", "settings")
    hazard_damage = _dig(settings, "hazardDamagePerTurn", default=None)
    if hazard_damage is None:
        hazard_damage = _dig(settings, "hazard_damage_per_turn", default=14)

    timeout_ms = _dig(state, "game", "timeout", default=500)
    constrictor = ruleset == "constrictor"  # cauda nunca sai do lugar, sem comida

    ctx = Context(
        width=board.width,
        height=board.height,
        turn=_dig(state, "turn", default=0),
        wrapped=ruleset == "wrapped",       # bordas "dão a volta"
        constrictor=constrictor,
        hazard_damage=int(hazard_damage),
        map_name=str(_dig(state, "game", "map", default="")),
        my_head=my_body[0],
        my_tail=my_body[-1],
        my_body=my_body,
        my_len=len(my_body),
        my_health=_dig(you, "health", default=MAX_HEALTH),
        enemies=enemies,
        food={_pt(c) for c in _dig(board, "food", default=[])},
        hazards={_pt(c) for c in _dig(board, "hazards", default=[])},
        free_at=_build_free_at(bodies, constrictor),
        deadline=started + (timeout_ms / 1000.0) * TUNING["time_budget"],
    )
    ctx.max_enemy_len = max((e.length for e in enemies), default=0)
    ctx.phase = get_phase(ctx)
    mods = PHASE_MODS.get(ctx.phase, {})
    ctx.w = {k: v * mods.get(k, 1.0) for k, v in WEIGHTS.items()}
    ctx.urgency = health_urgency(ctx)

    # Rival colado numa comida pode comer neste turno: a cauda dele NÃO sai do lugar.
    # Tratamos a casa da cauda como ocupada no turno 1 (cuidado conservador).
    if not constrictor:
        for enemy in enemies:
            if enemy.length > 1 and any(c in ctx.food for c in neighbors(ctx, enemy.head)):
                tail = enemy.body[-1]
                ctx.free_at[tail] = max(ctx.free_at.get(tail, 0), 2)

    for enemy in enemies:
        enemy.next_cells = [c for c in neighbors(ctx, enemy.head) if ctx.free_at.get(c, 0) <= 1]
    ctx.enemy_arrival = _build_enemy_arrival(ctx)
    return ctx


def _build_free_at(bodies: list, constrictor: bool) -> dict:
    """
    Para cada casa ocupada: em que turno (contado após o NOSSO movimento = 1) ela fica livre.

    O segmento i de uma cobra de tamanho L sai do tabuleiro quando 'L - i' movimentos
    acontecem. Logo a cauda (i = L-1) libera no turno 1: pisar nela é seguro.
    Se a cobra acabou de comer, a cauda está duplicada (body[-1] == body[-2]); como pegamos o
    MAIOR valor entre segmentos sobrepostos, essa casa só libera no turno 2 — automático.
    Em 'constrictor' ninguém libera casa nenhuma.
    """
    free_at: dict = {}
    for body in bodies:
        length = len(body)
        for i, seg in enumerate(body):
            t = INF if constrictor else length - i
            if t > free_at.get(seg, 0):
                free_at[seg] = t
    return free_at


def _build_enemy_arrival(ctx: Context) -> dict:
    """Menor turno em que algum adversário chega a cada casa (usado em comida e território)."""
    arrival: dict = {}
    for enemy in ctx.enemies:
        for cell, t in _bfs(ctx, enemy.head, 0).items():
            cur = arrival.get(cell)
            if cur is None or t < cur[0] or (t == cur[0] and enemy.length > cur[1]):
                arrival[cell] = (t, enemy.length)
    return arrival


def get_phase(ctx: Context) -> str:
    if ctx.turn < TUNING["opening_turns"]:
        return "opening"
    if len(ctx.enemies) == 1:
        return "endgame"
    return "midgame"


# --------------------------------------------------------------------------- #
# GEOMETRIA
# --------------------------------------------------------------------------- #

def get_next_position(ctx: Context, pos: Pt, move: str):
    """Casa resultante de um movimento; None se sair do tabuleiro (exceto em 'wrapped')."""
    dx, dy = MOVES[move]
    x, y = pos[0] + dx, pos[1] + dy
    if ctx.wrapped:
        return (x % ctx.width, y % ctx.height)
    if 0 <= x < ctx.width and 0 <= y < ctx.height:
        return (x, y)
    return None


def neighbors(ctx: Context, pos: Pt):
    for move in MOVE_ORDER:
        nxt = get_next_position(ctx, pos, move)
        if nxt is not None:
            yield nxt


def distance(ctx: Context, a: Pt, b: Pt) -> int:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if ctx.wrapped:
        dx, dy = min(dx, ctx.width - dx), min(dy, ctx.height - dy)
    return dx + dy


def _open_neighbors(ctx: Context, pos: Pt, turn: int) -> int:
    return sum(1 for n in neighbors(ctx, pos) if ctx.free_at.get(n, 0) <= turn)


# --------------------------------------------------------------------------- #
# MOVIMENTOS POSSÍVEIS / SEGURANÇA
# --------------------------------------------------------------------------- #

def health_after_move(ctx: Context, pos: Pt) -> int:
    """Vida após entrar em 'pos'. Comer devolve vida cheia (e não custa vida, ver /rules)."""
    if pos in ctx.food and not ctx.constrictor:
        return MAX_HEALTH
    health = ctx.my_health - 1
    if pos in ctx.hazards:
        health -= ctx.hazard_damage
    return health


def is_position_safe(ctx: Context, pos, turn: int = 1) -> bool:
    """Casa legal: dentro do tabuleiro, livre no turno 'turn' e sem morte por fome/hazard."""
    if pos is None:
        return False
    if ctx.free_at.get(pos, 0) > turn:
        return False
    return health_after_move(ctx, pos) > 0


def get_possible_moves(ctx: Context) -> list[str]:
    moves = []
    for move in MOVE_ORDER:
        if is_position_safe(ctx, get_next_position(ctx, ctx.my_head, move)):
            moves.append(move)
    return moves


def emergency_move(ctx: Context) -> str:
    """
    Nenhum movimento passa nos filtros. Escolhe o "menos pior", de forma determinística:
    1) dentro do tabuleiro, 2) sobrevive à vida, 3) casa que libera mais cedo
    (caso minha estimativa de cauda esteja errada, esse é o que ainda pode dar certo).
    """
    def rank(move: str):
        pos = get_next_position(ctx, ctx.my_head, move)
        if pos is None:
            return (3, INF)
        return (0 if health_after_move(ctx, pos) > 0 else 1, ctx.free_at.get(pos, 0))

    return min(MOVE_ORDER, key=rank)


# --------------------------------------------------------------------------- #
# BFS (turnos) e DIJKSTRA (custo em vida), ambos com liberação de caudas no tempo
# --------------------------------------------------------------------------- #

def _bfs(ctx: Context, start: Pt, start_turn: int) -> dict:
    """BFS: casa -> turno de chegada. Uma casa só é atravessável se já estiver livre nesse turno."""
    arrival = {start: start_turn}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        t = arrival[cell] + 1
        for nxt in neighbors(ctx, cell):
            if nxt in arrival or ctx.free_at.get(nxt, 0) > t:
                continue
            arrival[nxt] = t
            queue.append(nxt)
    return arrival


def flood_fill(ctx: Context, start: Pt, start_turn: int = 1) -> FloodResult:
    """Espaço acessível a partir de 'start' (nossa cabeça após o movimento = turno 1)."""
    arrival = _bfs(ctx, start, start_turn)
    tail_ok = (
        not ctx.constrictor
        and ctx.my_len >= 2
        and ctx.my_tail in arrival
    )
    return FloodResult(arrival=arrival, count=len(arrival), tail_reachable=tail_ok)


def step_cost(ctx: Context, cell: Pt) -> int:
    """Vida gasta ao ENTRAR numa casa: 1 por turno + dano extra se for hazard."""
    return 1 + (ctx.hazard_damage if cell in ctx.hazards else 0)


def dijkstra_routes(ctx: Context, start: Pt, start_turn: int = 1) -> dict:
    """
    Dijkstra a partir de 'start': casa -> (custo em vida, turno de chegada).
    O custo é medido em PONTOS DE VIDA, então dá para comparar direto com a saúde
    (um caminho por hazard custa muito mais do que o número de passos).
    O turno acompanha o caminho escolhido e serve para respeitar a liberação das caudas.
    """
    best = {start: (0, start_turn)}
    heap = [(0, start_turn, start)]
    while heap:
        cost, turn, cell = heapq.heappop(heap)
        if best[cell] != (cost, turn):
            continue  # entrada obsoleta
        t = turn + 1
        for nxt in neighbors(ctx, cell):
            if ctx.free_at.get(nxt, 0) > t:
                continue
            cand = (cost + step_cost(ctx, nxt), t)
            if nxt not in best or cand < best[nxt]:
                best[nxt] = cand
                heapq.heappush(heap, (cand[0], cand[1], nxt))
    return best


# --------------------------------------------------------------------------- #
# COMPONENTES DE PONTUAÇÃO (cada um devolve pontos: positivo = bom)
# --------------------------------------------------------------------------- #

def evaluate_space(ctx: Context, flood: FloodResult) -> float:
    """Espaço acessível + penalidade de beco (região menor que o corpo)."""
    comfort = max(2 * ctx.my_len, TUNING["space_comfort_min"])
    score = ctx.w["space"] * min(1.0, flood.count / comfort)
    if flood.count < ctx.my_len:
        penalty = ctx.w["dead_end"] * (1.0 - flood.count / ctx.my_len)
        if flood.tail_reachable:  # a cauda vai abrindo espaço: risco menor
            penalty *= 0.5
        score -= penalty
    return score


def evaluate_territory(ctx: Context, flood: FloodResult) -> float:
    """Fatia do tabuleiro que alcançamos antes (ou empatando sendo maiores) dos adversários."""
    mine = 0
    for cell, t in flood.arrival.items():
        enemy = ctx.enemy_arrival.get(cell)
        if enemy is None or t < enemy[0] or (t == enemy[0] and ctx.my_len > enemy[1]):
            mine += 1
    return ctx.w["territory"] * mine / (ctx.width * ctx.height)


def evaluate_mobility(ctx: Context, pos: Pt) -> float:
    """Quantas saídas existirão a partir da nova cabeça (turno 2)."""
    return ctx.w["mobility"] * _open_neighbors(ctx, pos, 2)


def evaluate_tail(ctx: Context, flood: FloodResult) -> float:
    """Cauda alcançável = rota de fuga. Se o espaço está apertado e não há fome, seguir a cauda."""
    if not flood.tail_reachable:
        return 0.0
    score = ctx.w["tail_reach"]
    comfort = max(2 * ctx.my_len, TUNING["space_comfort_min"])
    tightness = max(0.0, 1.0 - flood.count / (1.5 * comfort))  # 0 = folgado, 1 = apertado
    if ctx.my_len >= TUNING["tail_min_len"] or tightness > 0.5:
        d = flood.arrival[ctx.my_tail] - 1
        score += ctx.w["tail_follow"] * tightness * (1.0 - ctx.urgency) / (1 + d)
    return score


def health_urgency(ctx: Context) -> float:
    """0 = vida confortável, 1 = crítica (interpolação linear entre os dois limiares)."""
    lo, hi = TUNING["health_critical"], TUNING["health_comfort"]
    if ctx.hazards:
        lo += TUNING["hazard_urgency_shift"]
        hi += TUNING["hazard_urgency_shift"]
    h = ctx.my_health
    if h <= lo:
        return 1.0
    if h >= hi:
        return 0.0
    return (hi - h) / (hi - lo)


def evaluate_food(ctx: Context, pos: Pt, flood: FloodResult, routes: dict) -> float:
    """
    Valor da MELHOR comida (não da mais próxima) a partir da nova posição.
    A distância vem do Dijkstra (custo em vida: hazard pesa). Cada comida é descontada se:
    perderemos a corrida (comparada em TURNOS), está em hazard (sem urgência) ou fica em beco.
    O interesse base sobe com a urgência de vida.
    """
    if not ctx.food or ctx.constrictor:
        return 0.0

    best = 0.0
    for f in ctx.food:
        route = routes.get(f)
        if route is None:
            continue  # inalcançável a partir daqui
        cost, t = route
        value = 1.0 / (1 + cost)  # cost 0 = a comida está na casa do movimento

        rival = ctx.enemy_arrival.get(f)
        if rival is not None and (rival[0] < t or (rival[0] == t and rival[1] >= ctx.my_len)):
            value *= TUNING["contested"]
        if f in ctx.hazards and ctx.urgency < 0.8:
            value *= TUNING["hazard_food"]
        if _open_neighbors(ctx, f, t + 1) <= 1:
            value *= TUNING["deadend_food"]
        best = max(best, value)

    ahead = ctx.my_len > ctx.max_enemy_len + 1
    base = TUNING["food_base_ahead"] if ahead else TUNING["food_base"]
    if len(ctx.food) <= TUNING["scarce_food"]:
        base = min(1.0, base + 0.15)
    mix = base + (1.0 - base) * ctx.urgency
    return ctx.w["food"] * mix * best


def evaluate_health(ctx: Context, pos: Pt, flood: FloodResult, routes: dict) -> float:
    """Penaliza vida baixa e o cenário 'não chego em nenhuma comida a tempo'."""
    if pos in ctx.food and not ctx.constrictor:
        return 0.0
    h = health_after_move(ctx, pos)
    penalty = 0.0
    low = TUNING["low_health"]
    if h < low:
        penalty += ctx.w["low_health"] * (low - h) / low
    if not ctx.constrictor and ctx.food:
        # custo em vida até a comida mais barata: se >= vida restante, morremos de fome antes
        costs = [routes[f][0] for f in ctx.food if f in routes]
        if costs and min(costs) >= h:
            penalty += ctx.w["starvation"]
        elif not costs and h <= 30:
            penalty += ctx.w["starvation"] / 2
    return -penalty


def evaluate_hazards(ctx: Context, pos: Pt) -> float:
    """Custo de entrar em hazard; maior quanto menos vida sobraria depois do dano."""
    if pos not in ctx.hazards:
        return 0.0
    if pos in ctx.food:
        return -0.25 * ctx.w["hazard"]  # comer reabastece a vida
    risk = min(1.0, max(0.0, 1.0 - health_after_move(ctx, pos) / MAX_HEALTH))
    return -ctx.w["hazard"] * (1.0 + 2.0 * risk)


def evaluate_head_to_head(ctx: Context, pos: Pt) -> float:
    """
    Para cada adversário que PODE entrar em 'pos' no próximo turno:
      maior  -> perdemos (h2h_loss); igual -> ambos morrem (h2h_tie);
      menor  -> bônus de kill, dividido pelo nº de opções dele (se só tem 1 saída, é quase certo).
    (Se 'pos' tem comida, os dois comeriam e crescem juntos: a comparação de tamanho não muda.)
    """
    score = 0.0
    for e in ctx.enemies:
        if pos not in e.next_cells:
            continue
        if e.length > ctx.my_len:
            score -= ctx.w["h2h_loss"]
        elif e.length == ctx.my_len:
            score -= ctx.w["h2h_tie"]
        else:
            score += ctx.w["kill"] / max(1, len(e.next_cells))
    return score


def evaluate_enemies(ctx: Context, pos: Pt) -> float:
    """Pressão de proximidade: foge de cabeças maiores/iguais, persegue menores se saudável."""
    score = 0.0
    for e in ctx.enemies:
        d = distance(ctx, pos, e.head)
        if e.length >= ctx.my_len:
            if 1 <= d <= TUNING["threat_range"]:
                score -= ctx.w["threat"] / d
        elif d <= TUNING["hunt_range"] and ctx.urgency < 0.5:
            score += ctx.w["hunt"] / max(1, d)
    return score


def evaluate_position(ctx: Context, pos: Pt) -> float:
    """Desempate posicional: centro bom, borda ruim (não se aplica a 'wrapped')."""
    if ctx.wrapped:
        return 0.0
    cx, cy = (ctx.width - 1) / 2, (ctx.height - 1) / 2
    max_d = cx + cy
    center = 1.0 - (abs(pos[0] - cx) + abs(pos[1] - cy)) / max_d if max_d > 0 else 1.0
    score = ctx.w["center"] * center * (1.0 - ctx.urgency)
    if pos[0] in (0, ctx.width - 1) or pos[1] in (0, ctx.height - 1):
        score -= ctx.w["edge"]
    return score


def lookahead_adjustment(ctx: Context, move: str, pos: Pt, flood: FloodResult) -> float:
    """
    GANCHO para lookahead futuro (hoje devolve 0). Aqui entraria: simular o movimento,
    simular as respostas dos adversários e reavaliar.
    """
    return 0.0


def evaluate_move(ctx: Context, move: str) -> MoveEvaluation:
    pos = get_next_position(ctx, ctx.my_head, move)
    flood = flood_fill(ctx, pos)
    routes = dijkstra_routes(ctx, pos)
    parts = {
        "space": evaluate_space(ctx, flood),
        "territory": evaluate_territory(ctx, flood),
        "mobility": evaluate_mobility(ctx, pos),
        "tail": evaluate_tail(ctx, flood),
        "food": evaluate_food(ctx, pos, flood, routes),
        "health": evaluate_health(ctx, pos, flood, routes),
        "hazards": evaluate_hazards(ctx, pos),
        "h2h": evaluate_head_to_head(ctx, pos),
        "enemies": evaluate_enemies(ctx, pos),
        "position": evaluate_position(ctx, pos),
        "lookahead": lookahead_adjustment(ctx, move, pos, flood),
    }
    return MoveEvaluation(move=move, pos=pos, score=sum(parts.values()), parts=parts)


def _quick_score(ctx: Context, move: str) -> float:
    """Avaliação barata, usada só se o orçamento de tempo estourar."""
    pos = get_next_position(ctx, ctx.my_head, move)
    return 10.0 * _open_neighbors(ctx, pos, 2) + evaluate_head_to_head(ctx, pos)


# --------------------------------------------------------------------------- #
# FALLBACK — só roda se a lógica principal lançar exceção
# --------------------------------------------------------------------------- #

def _legacy_safe_moves(state: GameState) -> list[str]:
    """Paredes + todos os corpos (nossos e dos rivais). Respeita o modo 'wrapped'."""
    wrapped = str(_dig(state, "game", "ruleset", "name", default="")).lower() == "wrapped"
    w, h = state.board.width, state.board.height
    occupied = {_pt(c) for c in state.you.body}
    for snake in _dig(state.board, "snakes", default=[]):
        occupied |= {_pt(c) for c in snake.body}
    hx, hy = _pt(state.you.body[0])
    safe = []
    for name, (dx, dy) in MOVES.items():
        x, y = hx + dx, hy + dy
        if wrapped:
            x, y = x % w, y % h
        if 0 <= x < w and 0 <= y < h and (x, y) not in occupied:
            safe.append(name)
    return safe


# --------------------------------------------------------------------------- #
# PONTO DE ENTRADA
# --------------------------------------------------------------------------- #

def get_move(state: GameState) -> MoveResponse:
    started = time.perf_counter()
    try:
        ctx = build_context(state, started)
        possible_moves = get_possible_moves(ctx)

        if not possible_moves:
            move = emergency_move(ctx)
            logger.info("MOVE %d: sem saída! emergência -> %s", state.turn, move)
            return MoveResponse(move=move)

        scored: list[tuple[float, str]] = []
        for move in possible_moves:
            if time.perf_counter() > ctx.deadline:
                scored.append((_quick_score(ctx, move), move))
                continue
            ev = evaluate_move(ctx, move)
            logger.debug("MOVE %d %s: %.1f %s", state.turn, move, ev.score,
                         {k: round(v, 1) for k, v in ev.parts.items()})
            scored.append((ev.score, move))

        best_score = max(s for s, _ in scored)
        best_moves = [m for s, m in scored if s >= best_score - 1e-6]
        chosen = best_moves[0] if len(best_moves) == 1 else random.choice(best_moves)  # só desempate

        logger.info("MOVE %d [%s]: %s (score %.1f, %.0f ms)", state.turn, ctx.phase, chosen,
                    best_score, (time.perf_counter() - started) * 1000)
        return MoveResponse(move=chosen)

    except Exception:  # nunca devolver erro HTTP: um movimento ruim é melhor que nenhum
        logger.exception("MOVE %s: erro na lógica, usando fallback", getattr(state, "turn", "?"))
        safe = _legacy_safe_moves(state)
        return MoveResponse(move=safe[0] if safe else "up")
