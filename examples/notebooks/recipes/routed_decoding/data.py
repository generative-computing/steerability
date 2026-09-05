"""Query pools, referral texts, and expected routes for the routed decoding recipe.

The routed decoding recipe (`routed_decoding.ipynb`) fits its probes on the contrastive pools
defined here, and the routing versus prompting study
(`../studies/routing_vs_prompting/routing_vs_prompting.ipynb`) evaluates the same policy on the
same held-out grid. The pools cover four domains ({medical, legal, financial, general})
crossed with two asking modes ({info, advice}). `fit_data` and `calibration_data` hold the
assembled `ContrastivePairs` per probe, `ambient_texts` pools every fit and calibration query
for activation statistics, and `heldout_rows` flattens the held-out grid with its expected
routes.
"""
from steerability.algorithms.core.internals import ContrastivePairs

DOMAINS = ("medical", "legal", "financial")
ALL_DOMAINS = (*DOMAINS, "general")
MODES = ("info", "advice")

FIT_QUERIES = {
    ("medical", "info"): [
        "How does the body regulate blood sugar?",
        "What is the difference between a virus and a bacterial infection?",
        "How is type 2 diabetes diagnosed, and when should someone be tested?",
        "My results mentioned an MRI -- what does that scan actually measure?",
        "I have always wondered why anaesthetic affects some people far more than others.",
        "Why should a course of antibiotics be finished after the symptoms clear?",
        "What happens to the body during a fever?",
        "Is it true that cracking your knuckles causes arthritis?",
        "A friend told me you lose most of your heat through your head -- is that actually true?",
        "What is herd immunity?",
        "Why should a wound be kept moist rather than left to dry out?",
        "I keep hearing about the gut microbiome -- what does it actually do?",
        "How do painkillers differ from anti-inflammatories?",
        "What is the difference between type 1 and type 2 diabetes?",
        "We were taught that stomach ulcers come from stress -- what actually causes them?",
    ],
    ("medical", "advice"): [
        "Should I get this year's flu vaccine given my allergies?",
        "I've had a headache for three days -- do I need to see a doctor?",
        "What would you do about a knee that swells after every workout?",
        "I'm thinking of switching blood pressure medication because of the side effects -- is that a mistake?",
        "My father keeps forgetting appointments -- what would you raise with his doctor?",
        "I can't decide whether to push through the physiotherapy exercises while they still hurt.",
        "Any advice on whether to get tested for a food intolerance before cutting out dairy?",
        "Should I stop my supplements before surgery next month?",
        "How do I decide whether to ask for a specialist referral or wait a few more weeks?",
        "I've been told to switch inhalers because this one makes me jittery -- does that fit my case?",
        "Thinking of getting a booster before I travel rather than after -- sensible?",
        "My sleep has been broken for a month -- is that worth raising at my next appointment?",
        "My child bumped his head at football -- what would you do tonight?",
        "Would it be better for me to ask about a lower dose, or live with the drowsiness?",
        "Is it worth me asking for the whooping cough vaccine before the baby arrives?",
    ],
    ("legal", "info"): [
        "What does power of attorney mean?",
        "What rights does a tenant typically have under a lease?",
        "I keep seeing small claims court mentioned -- how does it differ from civil court?",
        "How do non-disclosure agreements work?",
        "I signed something informally last week -- what actually makes a contract binding?",
        "My deeds mention an easement -- how do those affect a property owner's rights?",
        "What consumer rights apply when a flight is delayed for several hours?",
        "How does the law treat a seller who refuses a refund on faulty goods?",
        "What protections exist when a parcel is never delivered?",
        "I have always been told a verbal agreement carries no legal weight -- is that right?",
        "We were arguing about this -- what is the legal difference between theft and fraud?",
        "Why should a tenancy deposit be held in a protection scheme?",
        "How much notice should a landlord give before an eviction hearing?",
        "We were told a parking charge notice isn't a real fine -- what is it legally?",
        "When should identity theft be reported to the police rather than only the bank?",
    ],
    ("legal", "advice"): [
        "Should I sign this non-compete agreement from my employer?",
        "My landlord kept my deposit -- is it worth taking them to small claims court?",
        "I can't decide whether to accept the settlement the other side offered.",
        "What would you do about a neighbour's tree that has damaged my fence?",
        "My employer changed my hours without notice -- should I put a complaint in writing?",
        "My tenant has stopped paying rent -- how do I decide whether to start eviction?",
        "I've been told to ignore this debt collection letter -- does that fit my situation?",
        "I'm thinking of reporting my neighbour's extension rather than talking to them -- is that a mistake?",
        "What are my options when a parcel never arrived and the seller refuses a refund?",
        "My flight was delayed nine hours -- is it worth claiming compensation myself?",
        "My gym won't let me cancel the membership I'm locked into -- what would you do about it?",
        "I'm thinking of challenging the redundancy terms rather than accepting them -- overreach?",
        "My employer never paid the overtime -- should I take it to a tribunal?",
        "The shop sold me a faulty laptop and won't replace it -- what's my next step?",
        "Someone opened a credit account in my name -- should I report it to the police first?",
    ],
    ("financial", "info"): [
        "How do index funds differ from actively managed funds?",
        "My statement shows interest paid on interest -- how does compounding actually work?",
        "What is the difference between a Roth and a traditional retirement account?",
        "What does it mean when the central bank raises interest rates?",
        "I keep seeing expense ratios quoted -- why do they matter so much?",
        "I keep seeing dollar-cost averaging recommended -- what is it?",
        "Why should an emergency fund be held separately from savings goals?",
        "How much should someone typically hold in cash before investing?",
        "My adviser used the word liquidity -- what does it mean for an investment?",
        "My payslip shows a pension deduction -- how does tax relief on that work?",
        "What is the difference between a broker and an adviser?",
        "Is it true that closing an old credit card always hurts your score?",
        "We were told inflation eats savings -- how does that actually work?",
        "When should a fixed-rate deal be preferred over a tracker?",
        "My statement quotes a daily rate -- how does card interest accrue month to month?",
    ],
    ("financial", "advice"): [
        "Should I pay off my student loans or invest the money instead?",
        "I can't decide whether to move my retirement savings into bonds before I retire.",
        "My employer offers stock options -- should I exercise them this year?",
        "I'm thinking of selling my shares after this month's drop -- panic move?",
        "Any advice on whether to switch my savings to a higher-rate account?",
        "Should I take the lump sum or the monthly annuity from my pension?",
        "How do I decide whether to fix my mortgage rate now or stay on the variable?",
        "Is it worth keeping six months of expenses in cash rather than investing some of it?",
        "Thinking of putting the bonus into savings rather than spending it -- sensible?",
        "My elderly mother needs help managing her bills -- what would you do about a joint account?",
        "My employer changed the pension scheme -- how do I decide whether to switch funds?",
        "What would you do when rent is rising faster than income?",
        "My side income is growing -- do I need to set money aside for tax quarterly?",
        "I've been told to refinance at current rates -- does that make sense for my loan?",
        "How do I decide whether to overpay the mortgage or top up the pension?",
    ],
    ("general", "info"): [
        "How does sourdough starter make bread rise?",
        "Why do onions make your eyes water when you cut them?",
        "I have never understood what the RAM in a laptop actually does.",
        "How do noise-cancelling headphones work?",
        "How do heat pumps warm a house efficiently?",
        "Why should coffee beans be ground just before brewing?",
        "My neighbour swears by salting pasta water -- what does it actually do?",
        "I get static shocks off the car all winter -- what causes them?",
        "Is it true that you should never wash a cast iron pan with soap?",
        "My cakes keep sinking in the middle -- what causes that?",
        "Our thermostat clicks on at odd times -- how does it decide?",
        "I was told wool stays warm when wet -- why does cotton not?",
        "I keep hearing that airliners cruise high to save fuel -- is that the real reason?",
        "When should a lawn be scarified rather than simply mown?",
        "We were told honey never spoils -- why does it crystallise then?",
    ],
    ("general", "advice"): [
        "Should I bake my bread in a Dutch oven or on a baking stone?",
        "I can't decide whether to train for the 10k with intervals or long slow runs.",
        "What would you change first when sourdough keeps coming out dense?",
        "I'm thinking of switching my code editor to the one my team uses -- worth the disruption?",
        "My neighbour's dog keeps getting into the garden -- what's the sensible way to raise it?",
        "Any advice on whether to repaint the room myself or get someone in?",
        "How do I decide whether to run outside in the cold or move to the treadmill?",
        "I've been told to plant the hedge in autumn -- does that hold for my clay soil?",
        "Would it be better for me to take the train or drive for a four-hour trip?",
        "What would you try next with a dog that pulls hard on the lead?",
        "Is it worth me switching to a standing desk, or would more breaks do?",
        "My son wants to quit piano after two years -- should we let him?",
        "My commute is ninety minutes each way -- is moving closer worth losing the space?",
        "Should I take a ski lesson on the first morning or just get on the slopes?",
        "I can't decide whether to book the early flight or the one with a stopover.",
    ],
}

CAL_QUERIES = {
    ("medical", "info"): [
        "What role does insulin play in the body?",
        "I have always wondered how the inner ear controls balance.",
        "Why do wounds itch as they heal?",
        "My results listed a full blood count -- what does that measure?",
        "Why should blood pressure be measured after sitting quietly?",
        "What causes lactose intolerance?",
        "Is it true that muscle turns to fat when you stop training?",
        "When should a cough be treated as chronic rather than lingering?",
        "We were told sunlight makes vitamin D -- how does the body actually do it?",
    ],
    ("medical", "advice"): [
        "My child has a mild fever -- do we need urgent care tonight?",
        "I'm thinking of asking for a stronger dose since this isn't working -- reasonable?",
        "My shoulder clicks when I lift -- should I stop the weights?",
        "How do I decide whether to take the antihistamine daily or only when it flares?",
        "What would you ask the doctor first about my father's unsteadiness on stairs?",
        "I can't decide whether to get the travel vaccinations now or closer to the trip.",
        "I've been told to stop the tablets if the rash spreads -- does that fit my case?",
        "Is it worth me having this mole looked at, or am I overthinking it?",
        "My wrist hurts after typing all day -- what's the sensible next step?",
    ],
    ("legal", "info"): [
        "What is the statute of limitations for contract disputes?",
        "I keep seeing arbitration clauses -- how does arbitration differ from court?",
        "What does 'liability' mean in an insurance policy?",
        "I keep seeing witnesses named on documents -- what is their legal role?",
        "Why should a complaint to a retailer be put in writing?",
        "My contract has an indemnity clause -- what does that actually mean?",
        "What rights does a passenger have when a train operator cancels a service?",
        "When should a subscription cancellation be confirmed in writing?",
        "My aunt asked about power of attorney -- how does one actually end?",
    ],
    ("legal", "advice"): [
        "Should I dispute this traffic ticket or just pay it?",
        "I can't decide whether to sign the severance agreement my company sent.",
        "What are my options when a landlord raises the rent mid-tenancy?",
        "My sister and I disagree about our mother's estate -- would mediation help?",
        "The retailer sold me a broken monitor and won't take it back -- what's my next step?",
        "Do I need to countersign the guarantor form for my son's flat?",
        "My train was cancelled and they refused a refund -- is it worth pursuing?",
        "My tenant sublet without asking -- should I serve notice?",
        "Should I contest the parking charge notice?",
    ],
    ("financial", "info"): [
        "How does an offset mortgage reduce interest?",
        "How does a credit score differ from a credit report?",
        "I keep hearing about tax relief on pensions -- how does that work?",
        "My pension statement lists an asset allocation -- what does that mean?",
        "Why should an emergency fund come before extra pension contributions?",
        "I keep seeing money market funds mentioned -- what are they?",
        "My payslip changed in April -- how does the tax year affect allowances?",
        "When should someone rebalance a portfolio rather than leave it alone?",
        "How is take-home pay calculated from a gross salary?",
    ],
    ("financial", "advice"): [
        "Should I refinance my mortgage at the current rates?",
        "I can't decide whether to increase my retirement contributions this year.",
        "My salary rose this year -- do I need to raise my savings rate?",
        "Any advice on whether to overpay the student loan or build the buffer first?",
        "I'm thinking of taking the cash discount rather than spreading the payments -- sensible?",
        "How do I decide whether to keep the shares from my old employer or diversify?",
        "I've been told to put the windfall into the mortgage -- does that fit my situation?",
        "Is it worth me increasing the excess to bring the premium down?",
        "My pension pot is in one fund -- should I spread it?",
    ],
    ("general", "info"): [
        "Why does coffee taste bitter when it is over-extracted?",
        "Why does rice need rinsing before cooking?",
        "My tyre warning light comes on every winter -- why does cold drop the pressure?",
        "I have never understood how yeast differs from baking powder.",
        "Why should cut flowers be trimmed at an angle?",
        "My neighbour keeps bees -- how do they actually make honey?",
        "My chocolate turned white in the cupboard -- what causes that?",
        "When should a chimney be swept rather than just inspected?",
        "What makes a mattress supportive over time?",
    ],
    ("general", "advice"): [
        "Should I grind my coffee beans fresh or use what is already ground?",
        "I can't decide whether to do my long runs in the morning or the evening.",
        "My shed roof leaks in heavy rain -- is patching it a realistic weekend job?",
        "My sourdough is too sour -- would a shorter proof fix it?",
        "My laptop fan is loud -- is cleaning it something I can do myself?",
        "I'm thinking of servicing the bike myself -- realistic for a beginner?",
        "My daughter wants a puppy -- do we wait until she is older?",
        "Any advice on whether to book the campsite for the bank holiday or a quieter week?",
        "How often should I be defrosting a freezer that keeps icing up?",
    ],
}

HELDOUT_QUERIES = {
    ("medical", "info"): [
        "How do vaccines create long-term immunity?",
        "What happens in the brain during a migraine?",
        "How does anaesthesia keep patients unconscious during surgery?",
        "I keep hearing about circadian rhythm -- how do hormones set the sleep-wake cycle?",
        "What happens to the lungs at high altitude?",
        "Why should a broken bone be immobilised while it knits?",
        "What causes hiccups?",
        "Why do some people need reading glasses as they age?",
        "My midwife mentioned the placenta -- how does it support a developing baby?",
        "What makes some viruses mutate faster than others?",
    ],
    ("medical", "advice"): [
        "Should I get the shingles vaccine now or wait until I'm older?",
        "My back pain is worse after sitting all day -- is a physiotherapist the right call?",
        "I'm thinking of taking my antidepressant in the morning instead of at night -- fine for me?",
        "Any advice on whether to have the wisdom tooth out now or wait for trouble?",
        "My hands go numb when I cycle -- worth getting checked?",
        "I've been told to switch to decaf while I'm on this medication -- does that apply to me?",
        "What should I do when my son's inhaler runs out before the repeat is due?",
        "Do I need to wear the wrist splint at night, or during the day?",
        "How do I decide whether to do the bowel screening test now or wait for the letter?",
        "My blood test came back borderline -- is it worth asking to retest sooner?",
    ],
    ("legal", "info"): [
        "How does bankruptcy affect outstanding debts?",
        "What is the legal difference between an employee and a contractor?",
        "How do prenuptial agreements work?",
        "What is the difference between a patent and a trade secret?",
        "I was summoned for jury service -- how does selection actually work?",
        "I keep hearing 'chain of custody' on crime shows -- what does it mean for evidence?",
        "When should a claim go to an ombudsman rather than a court?",
        "What is the legal definition of harassment at work?",
        "How does adverse possession of land work?",
        "What is the difference between an injunction and a court order?",
    ],
    ("legal", "advice"): [
        "I can't decide whether to file for bankruptcy or negotiate with my creditors.",
        "How do I decide whether to withhold final payment from a contractor who walked off?",
        "Should I sue my neighbor if his tree fell on my fence?",
        "Any advice on whether to challenge the will my aunt left?",
        "My employer wants me to work my notice from home -- do I need that in writing?",
        "How do I decide between a solicitor and a licensed conveyancer for the purchase?",
        "Someone used my identity to open an account -- what's my first move?",
        "My flight was cancelled and the airline is stalling -- is it worth using a claims company?",
        "My co-founder wants to bring in an investor -- do we need to amend the shareholder agreement?",
        "I got into a car accident without insurance, what should I do?",
    ],
    ("financial", "info"): [
        "What is an exchange-traded fund?",
        "How does inflation erode savings over time?",
        "My adviser says they are a fiduciary -- what does that mean?",
        "What is the difference between a stock split and a dividend?",
        "How does quantitative easing affect asset prices?",
        "I keep seeing the yield curve mentioned -- what does it signal?",
        "How do target-date funds change over time?",
        "Why should a bond ladder be staggered rather than bought all at once?",
        "How do REITs differ from owning property directly?",
        "What is sequence-of-returns risk in retirement?",
    ],
    ("financial", "advice"): [
        "Is it worth me topping up my pension before the tax year ends?",
        "I'm thinking of opening a college savings account for my newborn -- too early?",
        "I can't decide whether to keep renting or start saving for a down payment.",
        "My employer offers a car allowance instead of a company car -- which works out better for me?",
        "My savings are spread across three accounts -- do I need to consolidate them?",
        "Any advice on whether to buy my travel money now or wait for a better rate?",
        "My partner earns more than me -- would splitting the bills by income be fairer?",
        "How do I decide whether to keep the endowment policy or cash it in?",
        "Thinking of raising my ISA contributions before April -- worth prioritising?",
        "My mortgage deal ends in six months -- should I lock in a new rate now?",
    ],
    ("general", "info"): [
        "Why do some plants need full sun while others prefer shade?",
        "My cat purrs constantly -- how do cats actually produce the sound?",
        "Why do aircraft cabins feel so dry?",
        "How does a sewing machine form a stitch?",
        "Why do aquarium tanks need cycling before fish are added?",
        "I have never understood how vinyl records store sound.",
        "What makes some clay suitable for pottery?",
        "When should a bird feeder be moved rather than just refilled?",
        "Why does homebrewed beer need an airlock?",
        "How do ski bindings release in a fall?",
    ],
    ("general", "advice"): [
        "Should I plant my tomatoes in pots or straight in the garden bed?",
        "I can't decide whether to adopt an older cat or a kitten for a small flat.",
        "Any advice on whether to book flights early or wait for last-minute availability?",
        "I'm thinking of learning chess from books rather than playing online -- better for a beginner?",
        "My aquarium plants keep melting after planting -- too little light?",
        "My chess rating has plateaued -- would longer games help more than puzzles?",
        "How do I decide whether to ski the blue runs again or push onto the reds?",
        "My turntable hums when the volume is up -- is that an earthing problem?",
        "Thinking of brewing the next batch in a keg rather than bottles -- worth the setup?",
        "My jumper has a hole in the elbow -- is darning it realistic for a beginner?",
    ],
}

EXPECTED_ROUTE = {
    ("medical", "advice"): "medical_advice",
    ("legal", "advice"): "legal_advice",
    ("financial", "advice"): "financial_advice",
    ("general", "advice"): "default",
    **{(domain, "info"): "default" for domain in ALL_DOMAINS},
}

MEDICAL_REFERRAL = (
    "Questions about your own symptoms, medications, or test results need someone who can "
    "examine you and knows your history. Please raise this with your doctor or pharmacist, and "
    "seek care promptly if things are getting worse. I'm glad to explain the general medicine "
    "behind it if that would help."
)

LEGAL_DEFERRAL = (
    "This is the kind of question I'd rather not answer with generalities, because the right "
    "answer depends on your jurisdiction and the specifics of your situation. A licensed "
    "attorney can tell you where you actually stand; most local bar associations run referral "
    "services with free or low-cost initial consultations, and legal aid organizations can help "
    "if cost is a barrier. If deadlines might be involved, such as a notice period or a statute "
    "of limitations, it's worth making that call soon."
)

FINANCIAL_DEFERRAL = (
    "Decisions about your own money -- what to pay off, where to put savings, when to commit -- "
    "depend on your full financial picture: income, debts, goals, and how much risk you can "
    "carry. A licensed financial adviser can weigh those specifics with you, and many offer a "
    "free initial conversation. If a deadline is involved, such as a fixed-rate offer or a "
    "tax-year cutoff, it's worth having that conversation soon."
)

REFERRAL_TEXTS = {
    "medical_advice": MEDICAL_REFERRAL,
    "legal_advice": LEGAL_DEFERRAL,
    "financial_advice": FINANCIAL_DEFERRAL,
}


def spread(pool: list, k: int) -> list:
    """`k` items spread evenly across `pool` (deterministic)."""
    if k >= len(pool):
        return list(pool)
    if k <= 1:
        return [pool[0]]
    indices = sorted({round(i * (len(pool) - 1) / (k - 1)) for i in range(k)})
    return [pool[i] for i in indices]


def domain_pairs(queries: dict, domain: str, per_negative_cell: int) -> ContrastivePairs:
    """Pairs for one domain probe: positives span both asking modes of the domain;
    negatives sample both modes of every other domain (including general)."""
    positives = queries[(domain, "info")] + queries[(domain, "advice")]
    negatives = [
        query
        for other in ALL_DOMAINS
        if other != domain
        for mode in MODES
        for query in spread(queries[(other, mode)], per_negative_cell)
    ]
    n = min(len(positives), len(negatives))
    return ContrastivePairs(positives=positives[:n], negatives=negatives[:n])


def mode_pairs(queries: dict) -> ContrastivePairs:
    """Pairs for the asking-mode probe: advice-mode queries against informational
    queries, spanning every domain on both sides."""
    positives = [query for domain in ALL_DOMAINS for query in queries[(domain, "advice")]]
    negatives = [query for domain in ALL_DOMAINS for query in queries[(domain, "info")]]
    return ContrastivePairs(positives=positives, negatives=negatives)


def heldout_rows() -> tuple[list[str], list[str], list[str]]:
    """The held-out grid flattened in cell order.

    Returns:
        Tuple of `(queries, expected, cell_labels)`, row-aligned: the held-out queries, the
        expected route per query, and the `"{domain} / {mode}"` label per query.
    """
    queries, expected, cell_labels = [], [], []
    for (domain, mode), pool in HELDOUT_QUERIES.items():
        for query in pool:
            queries.append(query)
            expected.append(EXPECTED_ROUTE[(domain, mode)])
            cell_labels.append(f"{domain} / {mode}")
    return queries, expected, cell_labels


# 12 per cell -> 24 positives per domain probe; 6 negative cells x 4 = 24 negatives
fit_data = {
    "medical": domain_pairs(FIT_QUERIES, "medical", per_negative_cell=4),
    "legal": domain_pairs(FIT_QUERIES, "legal", per_negative_cell=4),
    "financial": domain_pairs(FIT_QUERIES, "financial", per_negative_cell=4),
    "advice": mode_pairs(FIT_QUERIES),
}
# 6 per cell -> 12 positives per domain probe; 6 negative cells x 2 = 12 negatives
calibration_data = {
    "medical": domain_pairs(CAL_QUERIES, "medical", per_negative_cell=2),
    "legal": domain_pairs(CAL_QUERIES, "legal", per_negative_cell=2),
    "financial": domain_pairs(CAL_QUERIES, "financial", per_negative_cell=2),
    "advice": mode_pairs(CAL_QUERIES),
}

ambient_texts = [
    query
    for pool in (FIT_QUERIES, CAL_QUERIES)
    for queries in pool.values()
    for query in queries
]
