"""
Genera una base de datos SINTÉTICA de urgencias que respeta el diccionario de variables (Variables_Urgencia.pdf).

- Mismas columnas, tipos y códigos que la base real (separada por tabulador, fechas ISO).
- tri_presion_minima = presión SISTÓLICA y tri_presion_maxima = DIASTÓLICA (como indica el diccionario).
- tri_vb = se realizó electrocardiograma (S/N). riesgo_de_caida = código 1–4. sex_cod = M/F/I/D.
- pac_edad_tipo = A/M/D. glosa_tipo_atencion = ADULTO/ADONTOLOGICO.
- mot_glosa y dest_glosa usan valores del catálogo del diccionario.
- Los datos son coherentes entre sí: el motivo define la categoría probable, la categoría y el motivo definen
  los signos vitales, y la categoría define destino y pronóstico. Incluye algunos errores de registro típicos
  (vacíos, ceros, presiones invertidas, esperas negativas, duplicados) para probar la limpieza del EDA.

Los valores son inventados: sirven para probar el análisis, NO para sacar conclusiones.

Uso:
    python generar_datos_sinteticos.py                      # 50.000 atenciones, 2023-01-01 a 2024-12-31
    python generar_datos_sinteticos.py -n 20000 --salida prueba.txt --semilla 7
"""
import argparse

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------
# Escenarios clínicos: (peso, [mot_glosa del catálogo], [textos de tri_motivo_cons],
#                       probabilidad C1..C5, modificadores de signos vitales, reglas especiales)
# ------------------------------------------------------------------------------------------------
ESCENARIOS = {
    "toracico": dict(
        peso=6, mot=["DOLOR Y/O MOLESTIAS TORAX", "DOLOR DE PECHO"],
        txt=["DOLOR TORACICO OPRESIVO DE {h} HRS DE EVOLUCION, IRRADIADO A BRAZO IZQUIERDO",
             "PACIENTE REFIERE DOLOR EN EL PECHO DESDE HACE {h} HORAS, SE TOMA ECG",
             "DOLOR PRECORDIAL ASOCIADO A SUDORACION. ANTECEDENTE DE HTA",
             "DERIVADO DE SAPU POR DOLOR TORACICO, OB. SCA"],
        cat=[.04, .30, .50, .14, .02], vs=dict(pulso=8, pas=8, eva=2), ecg=.9),
    "abdominal": dict(
        peso=12, mot=["DOLOR Y/O MOLESTIAS ABDOMEN", "COLICO/ABDOMINAL", "COLICO/RENAL", "COLICO/BILIAR",
                      "DOLOR DE ESTOMAGO"],
        txt=["DOLOR ABDOMINAL DE {h} HRS DE EVOLUCION, ASOCIADO A NAUSEAS",
             "DOLOR EN FOSA ILIACA DERECHA DESDE AYER, SIN FIEBRE",
             "COLICO RENAL, DOLOR LUMBAR IRRADIADO A GENITALES",
             "DOLOR EN HIPOCONDRIO DERECHO POST INGESTA DE COMIDA GRASA",
             "EPIGASTRALGIA DE {d} DIAS, VOMITOS EN {n} OCASIONES"],
        cat=[.002, .05, .45, .45, .048], vs=dict(pulso=5, eva=3)),
    "respiratorio": dict(
        peso=13, mot=["PROBLEMAS RESPIRATORIOS", "SINDROME RESPIRATORIO", "TOS", "GRIPE", "ASMA",
                      "DOLOR DE GARGANTA"],
        txt=["TOS Y FIEBRE DE {d} DIAS DE EVOLUCION", "DIFICULTAD RESPIRATORIA, SIBILANCIAS. ASMATICO CONOCIDO",
             "ODINOFAGIA Y FIEBRE DESDE AYER", "DISNEA DE ESFUERZO PROGRESIVA, USUARIO DE OXIGENO DOMICILIARIO",
             "CUADRO GRIPAL, CONGESTION NASAL Y MIALGIAS", "TOS PRODUCTIVA, SIN DIFICULTAD RESPIRATORIA"],
        cat=[.01, .08, .36, .45, .10], vs=dict(fr=4, sato2=-3, temp=.6, pulso=8), invierno=2.0),
    "fiebre": dict(
        peso=5, mot=["FIEBRE", "FIEBRE Y OTROS"],
        txt=["FIEBRE CUANTIFICADA DE 39 GRADOS DESDE AYER", "FIEBRE Y CALOFRIOS, MALESTAR GENERAL",
             "FIEBRE DE {d} DIAS SIN FOCO CLARO"],
        cat=[.003, .05, .40, .47, .077], vs=dict(temp=1.3, pulso=12), invierno=1.4),
    "trauma": dict(
        peso=14, mot=["ACCIDENTE/TRAUMATISMO EN EL HOGAR", "ACCIDENTE/TRAUMATISMO VIA PUBLICA",
                      "ACCIDENTE/TRAUMATISMO OTRO TIPO", "CAIDA", "CONTUSION", "TORCEDURA", "FRACTURAS"],
        txt=["CAIDA A NIVEL, CONTUSION EN RODILLA DERECHA", "ESGUINCE DE TOBILLO JUGANDO FUTBOL",
             "TEC SIN PERDIDA DE CONCIENCIA, CAIDA DE ESCALERA", "ACCIDENTE DE TRANSITO, CONDUCTOR DE MOTO",
             "CAIDA DE ALTURA, DOLOR EN MUÑECA IZQUIERDA CON DEFORMIDAD", "TRAUMA DE MANO CON MARTILLO"],
        cat=[.008, .05, .30, .55, .092], vs=dict(eva=2), caida=1.5),
    "herida": dict(
        peso=5, mot=["HERIDAS Y/O HEMORRAGIA", "HERIDA"],
        txt=["HERIDA CORTANTE EN DEDO CON CUCHILLO", "HERIDA CORTOCONTUSA EN CUERO CABELLUDO",
             "EPISTAXIS DE {h} HRS QUE NO CEDE"],
        cat=[.005, .04, .30, .55, .105], vs=dict(eva=1)),
    "cefalea": dict(
        peso=4, mot=["DOLOR Y/O MOLESTIAS CABEZA Y CUELLO", "CEFALEA", "DOLOR DE CABEZA"],
        txt=["CEFALEA HOLOCRANEANA DE {h} HRS, NO CEDE CON PARACETAMOL", "CEFALEA INTENSA SUBITA, FOTOFOBIA",
             "CEFALEA Y VOMITOS, ANTECEDENTE DE MIGRAÑA"],
        cat=[.005, .08, .45, .42, .045], vs=dict(eva=3, pas=10)),
    "gi": dict(
        peso=6, mot=["VOMITOS Y/O DIARREA", "VOMITOS", "DIARREA"],
        txt=["DIARREA Y VOMITOS DESDE AYER, TOLERA LIQUIDOS", "DEPOSICIONES LIQUIDAS {n} VECES EN EL DIA",
             "VOMITOS PERSISTENTES, NO TOLERA VIA ORAL"],
        cat=[.002, .04, .38, .50, .078], vs=dict(pulso=8, temp=.3)),
    "hta": dict(
        peso=3, mot=["PRESION ARTERIAL ELEVADA", "HIPERTENSION"],
        txt=["PRESION ELEVADA EN CONTROL, CEFALEA", "HTA DESCOMPENSADA, NO TOMA SUS MEDICAMENTOS",
             "MAREO Y PRESION ALTA EN DOMICILIO"],
        cat=[.005, .12, .50, .35, .025], vs=dict(pas=40, pad=18), edad=(60, 14), ecg=.5),
    "diabetes": dict(
        peso=1.5, mot=["DIABETES MELLITUS DESCOMPENSADA", "DIABETES"],
        txt=["HIPERGLICEMIA EN DOMICILIO, POLIURIA Y POLIDIPSIA", "TRAIDA POR HIPOGLICEMIA EN DOMICILIO, USUARIA DE INSULINA",
             "DIABETICO DESCOMPENSADO, HGT HIGH EN DOMICILIO"],
        cat=[.02, .15, .50, .30, .03], vs=dict(), edad=(58, 15), hgt="alto", dm=.95),
    "neuro": dict(
        peso=3, mot=["CONVULSIONES", "EPILEPSIA", "MAREOS"],
        txt=["CRISIS CONVULSIVA TONICO CLONICA EN DOMICILIO, EPILEPTICO CONOCIDO", "SINCOPE EN VIA PUBLICA",
             "MAREO ROTATORIO CON VOMITOS", "DEBILIDAD DE BRAZO DERECHO Y DESVIACION DE COMISURA DESDE HACE {h} HRS"],
        cat=[.05, .20, .45, .27, .03], vs=dict(ggw=-1.5, pas=8), edad=(52, 22), ecg=.4, hgt="medir"),
    "gineco": dict(
        peso=4, mot=["MOLESTIAS GINECOLOGICAS", "EMBARAZO CON CONTRACCIONES", "MOLESTIAS DEL EMBARAZO",
                     "DOLOR/OVARIO", "DOLOR VAGINAL"],
        txt=["EMBARAZO DE {s} SEMANAS CON CONTRACCIONES", "GESTANTE DE {s} SEMANAS CON SANGRADO VAGINAL",
             "DOLOR PELVICO Y FLUJO VAGINAL", "METRORRAGIA ABUNDANTE"],
        cat=[.003, .10, .45, .40, .047], vs=dict(eva=2), sexo="F", edad=(28, 7)),
    "psiq_tox": dict(
        peso=2.5, mot=["INGESTA DE MEDICAMENTOS", "DEPRESION", "TOMA DE MUESTRA TOXICOLOGICA", "INGESTA DE CLORO"],
        txt=["INGESTA DE {n}0 COMPRIMIDOS DE CLONAZEPAM CON FINES SUICIDAS", "CRISIS DE PANICO, ANSIEDAD",
             "IDEACION SUICIDA, TRAIDA POR FAMILIAR", "AGITACION PSICOMOTORA, TRAIDO POR CARABINEROS"],
        cat=[.02, .20, .50, .25, .03], vs=dict(pulso=10, ggw=-.5), edad=(30, 12)),
    "agresion": dict(
        peso=3, mot=["AGRESION Y/O VIOLENCIA", "CONSTATACION DE LESIONES", "ALCOHOLEMIA",
                     "ALCOHOLEMIA + CONST.LESIONES", "VIF"],
        txt=["CONSTATACION DE LESIONES, TRAIDO POR CARABINEROS", "AGRESION POR TERCEROS, CONTUSION FACIAL",
             "ALCOHOLEMIA POR CONTROL VEHICULAR", "VIF, REFIERE GOLPES EN BRAZOS"],
        cat=[.003, .03, .25, .55, .167], vs=dict(), edad=(32, 12), carabineros=.5),
    "osteo": dict(
        peso=6, mot=["DOLOR Y/O MOLESTIAS EXTREMIDADES", "DOLOR LUMBAR", "DOLOR/ESPALDA", "DOLOR RODILLA"],
        txt=["LUMBAGO DE {d} DIAS SIN TRAUMA", "DOLOR EN RODILLA IZQUIERDA AL CAMINAR",
             "DOLOR CERVICAL POST ESFUERZO", "DOLOR EN PIERNA DERECHA Y EDEMA"],
        cat=[.001, .02, .25, .60, .129], vs=dict(eva=2)),
    "piel_alergia": dict(
        peso=3, mot=["ALERGIA", "PICADURA/INSECTO", "MORDEDURA ANIMAL", "QUEMADURAS", "CELULITIS"],
        txt=["RONCHAS EN EL CUERPO Y PRURITO POST INGESTA DE MARISCOS", "MORDEDURA DE PERRO EN PIERNA",
             "QUEMADURA CON AGUA HIRVIENDO EN MANO", "ERITEMA Y AUMENTO DE VOLUMEN EN PIERNA"],
        cat=[.005, .05, .30, .50, .145], vs=dict(temp=.2)),
    "orl_ocular": dict(
        peso=2.5, mot=["DOLOR DE OIDO", "DOLOR DE OJOS", "CUERPO EXTRANO OJOS", "CUERPO EXTRANO NARIZ"],
        txt=["OTALGIA DERECHA DE {d} DIAS", "OJO ROJO Y SECRECION", "CUERPO EXTRAÑO EN OJO IZQUIERDO TRABAJANDO"],
        cat=[.001, .02, .20, .60, .179], vs=dict()),
    "dental": dict(
        peso=2.5, mot=["DOLOR DE MUELAS"], txt=["ODONTALGIA DE {d} DIAS", "DOLOR DE MUELA CON AUMENTO DE VOLUMEN"],
        cat=[0, .005, .10, .55, .345], vs=dict(eva=3), dental=True),
    "urinario": dict(
        peso=3, mot=["DOLOR ORINAR"], txt=["DISURIA Y POLAQUIURIA DE {d} DIAS", "ARDOR AL ORINAR Y FIEBRE"],
        cat=[.001, .02, .30, .58, .099], vs=dict(temp=.3)),
    "critico": dict(
        peso=.6, mot=["OTROS, ESPECIFICAR", "PROBLEMAS RESPIRATORIOS", "CONVULSIONES"],
        txt=["TRAIDA POR SAMU POR OBS SHOCK SEPTICO. INGRESA CON MANEJO DE VIA AEREA AVANZADA",
             "PCR EXTRAHOSPITALARIA, INGRESA EN RCP", "COMPROMISO DE CONCIENCIA, GLASGOW 8, TRAIDO POR SAMU",
             "POLITRAUMATIZADO, ACCIDENTE DE TRANSITO DE ALTA ENERGIA"],
        cat=[.55, .40, .05, 0, 0], vs=dict(pulso=30, fr=8, sato2=-8, pas=-30, ggw=-6), samu=.9, ecg=.8, hgt="medir"),
    "otros": dict(
        peso=8, mot=["OTROS, ESPECIFICAR", "OTRO", "INTERCONSULTA", "DERIVADO", "OTROS DOLORES", "INFLAMACION",
                     "EDEMA /VARIOS"],
        txt=["MALESTAR GENERAL", "CONTROL POST OPERATORIO, DOLOR EN HERIDA", "SOLICITA CERTIFICADO",
             "DERIVADO DE CESFAM PARA EVALUACION", "DECAIMIENTO Y BAJA INGESTA DE ALIMENTOS",
             "RECETA DE MEDICAMENTOS"],
        cat=[.003, .04, .30, .50, .157], vs=dict()),
}

ALERGENOS = ["PENICILINA", "PNC", "PENISILINA", "AMOXICILINA", "AINES", "DIPIRONA", "METAMIZOL", "SULFAS",
             "ASPIRINA", "KETOROLACO", "IBUPROFENO", "TRAMADOL", "CEFALOSPORINAS", "MARISCOS", "LATEX",
             "YODO", "NO RECUERDA", "CLARITROMICINA"]
PESO_ALERGENO = np.array([20, 8, 3, 8, 10, 8, 4, 6, 4, 3, 4, 2, 2, 3, 2, 2, 4, 2], float)

MEDIOS = ["POR SUS PROPIOS MEDIOS", "AMBULANCIA APS", "SAMU MOVIL BASICO", "SAMU MOVIL AVANZADO", "CARABINEROS"]


def generar(n, inicio, fin, semilla):
    rng = np.random.default_rng(semilla)

    # ---------------- llegada: tendencia leve + invierno + día de semana + hora
    dias = pd.date_range(inicio, fin, freq="D")
    t = np.arange(len(dias)) / len(dias)
    mes = dias.month.to_numpy()
    invierno = np.isin(mes, [6, 7, 8]).astype(float)
    dow = dias.dayofweek.to_numpy()
    w_dia = (1 + 0.08 * t) * (1 + 0.15 * invierno) * np.array([1.12, 1.02, 1.0, 0.98, 1.0, 0.93, 0.95])[dow]
    dia_idx = rng.choice(len(dias), n, p=w_dia / w_dia.sum())
    hw = np.array([2, 1.5, 1.1, .9, .8, 1, 1.8, 3.5, 6, 7.5, 7.8, 7.3, 6.5, 6.2, 6.3, 6.2, 6.1, 6, 6, 5.6, 5, 4.4, 3.5, 2.6])
    hora = rng.choice(24, n, p=hw / hw.sum())
    urg = dias[dia_idx] + pd.to_timedelta(hora, "h") + pd.to_timedelta(rng.integers(0, 3600, n), "s")
    urg = urg.floor("min")
    inv_pac = invierno[dia_idx]

    # ---------------- escenario (más respiratorio y fiebre en invierno)
    nombres = list(ESCENARIOS)
    base = np.array([ESCENARIOS[k]["peso"] for k in nombres])
    boost = np.array([ESCENARIOS[k].get("invierno", 1.0) for k in nombres])
    p_ver = base / base.sum()
    p_inv = base * boost / (base * boost).sum()
    u = rng.random(n)
    esc = np.where(inv_pac == 1, np.searchsorted(np.cumsum(p_inv), u), np.searchsorted(np.cumsum(p_ver), u))
    esc = np.array(nombres)[np.minimum(esc, len(nombres) - 1)]

    # ---------------- edad y sexo
    edad = np.clip(np.where(rng.random(n) < .18, rng.gamma(1.4, 3.5, n), rng.gamma(2.6, 16, n) + 12), 0, 102)
    for k, e in ESCENARIOS.items():
        if "edad" in e:
            m = esc == k
            edad[m] = np.clip(rng.normal(*e["edad"], m.sum()), 14, 100)
    sexo = rng.choice(["M", "F", "I", "D"], n, p=[.465, .53, .003, .002])
    sexo[esc == "gineco"] = "F"

    # ---------------- categoría según escenario (+ los adultos mayores se categorizan algo más grave)
    cat = np.empty(n, int)
    for k, e in ESCENARIOS.items():
        m = esc == k
        cat[m] = rng.choice([1, 2, 3, 4, 5], m.sum(), p=np.array(e["cat"]) / sum(e["cat"]))
    mayor = (edad >= 75) & (cat >= 4) & (rng.random(n) < .25)
    cat[mayor] -= 1
    sev = 3 - cat   # >0 más grave, <0 más leve

    # ---------------- espera al triage (min): depende de categoría y hora punta
    punta = ((hora >= 10) & (hora <= 13)) | ((hora >= 18) & (hora <= 20))
    espera = rng.lognormal(np.log(12) + 0.25 * (cat - 3) + 0.35 * punta + 0.15 * inv_pac, 0.75)
    espera[cat == 1] = rng.exponential(2, (cat == 1).sum())
    tri = urg + pd.to_timedelta(np.round(espera), "m")

    # ---------------- signos vitales (adulto; ajuste simple para niños)
    def mod(clave):
        out = np.zeros(n)
        for k, e in ESCENARIOS.items():
            out[esc == k] = e["vs"].get(clave, 0)
        return out

    nino = edad < 12
    pulso = rng.normal(82 + 4 * sev + mod("pulso") + 30 * nino, 13)
    fr = rng.normal(17 + 1.2 * sev + mod("fr") + 8 * nino, 2.5)
    sato2 = np.clip(rng.normal(97.5 - 0.6 * np.maximum(sev, 0) + mod("sato2"), 1.4), 60, 100)
    temp = rng.normal(36.6 + mod("temp"), 0.45)
    pas = rng.normal(125 + 0.35 * (edad - 45) + mod("pas") - 25 * nino, 17)
    pad = rng.normal(76 + 0.4 * (pas - 125) * 0.5 + mod("pad") - 15 * nino, 9)
    pad = np.minimum(pad, pas - 15)
    ggw = np.clip(np.round(15 + np.minimum(rng.normal(mod("ggw") * (sev > 0), 0.6), 0)), 3, 15)
    eva = np.clip(np.round(rng.normal(3.5 + mod("eva") + 0.4 * sev, 2.2)), 0, 10)
    peso = np.where(nino, np.clip(3.5 + 2.6 * edad + rng.normal(0, 2, n), 2, 60),
                    np.clip(rng.normal(np.where(sexo == "M", 80, 68), 13), 35, 180))

    hgt_regla = np.array([ESCENARIOS[k].get("hgt", "") for k in esc])
    dm_prev = np.clip(0.02 + 0.004 * np.maximum(edad - 30, 0), 0, .35)
    dm = rng.random(n) < dm_prev
    dm[esc == "diabetes"] = rng.random((esc == "diabetes").sum()) < .95
    hgt = np.full(n, np.nan)
    medir_hgt = dm | (hgt_regla != "") | (rng.random(n) < .05)
    hgt[medir_hgt] = rng.normal(np.where(dm[medir_hgt], 190, 110), np.where(dm[medir_hgt], 60, 22))
    alto = hgt_regla == "alto"
    hgt[alto] = np.where(rng.random(alto.sum()) < .2, rng.normal(50, 10, alto.sum()), rng.normal(380, 90, alto.sum()))
    hgt = np.clip(hgt, 20, 600)

    # Completitud: menos signos vitales en C5, dental y niños pequeños
    p_falta = np.select([esc == "dental", cat == 5, cat == 4], [.7, .35, .12], .05)
    falta = rng.random(n) < p_falta

    def vacio(x, extra=0.0):
        x = np.round(x.astype(float), 1)
        x[falta | (rng.random(n) < extra)] = np.nan
        return x

    # ---------------- antecedentes S/N/D/vacío
    def snd(p_si, p_d=.04, p_vacio=.06):
        r = rng.random(n)
        v = np.where(r < p_si, "S", "N").astype(object)
        v[rng.random(n) < p_d] = "D"
        v[rng.random(n) < p_vacio] = None
        return v

    hta_prev = np.clip(0.03 + 0.008 * np.maximum(edad - 30, 0), 0, .6)
    hta_prev[esc == "hta"] = .85
    lcfa_prev = np.clip(0.005 + 0.003 * np.maximum(edad - 45, 0), 0, .2)
    tri_dm = snd(np.where(dm, 1, 0))
    tri_hta = snd(hta_prev)
    tri_lcfa = snd(lcfa_prev)
    p_ecg = np.array([ESCENARIOS[k].get("ecg", .05) for k in esc]) + .15 * (cat <= 2) + .1 * (edad >= 65)
    tri_vb = np.where(rng.random(n) < np.clip(p_ecg, 0, .98), "S", "N").astype(object)   # ECG: solo S/N
    tri_vb[rng.random(n) < .05] = None
    tri_alergia = snd(np.full(n, .11), p_d=.03, p_vacio=.05)
    detalle = np.full(n, None, dtype=object)
    con_det = (tri_alergia == "S") & (rng.random(n) < .9)
    detalle[con_det] = rng.choice(ALERGENOS, con_det.sum(), p=PESO_ALERGENO / PESO_ALERGENO.sum())
    sin_alergia_con_det = (tri_alergia == "N") & (rng.random(n) < .01)    # error de registro típico
    detalle[sin_alergia_con_det] = "NIEGA"

    # ---------------- riesgo de caída: código 1–4 (más alto en mayores y trauma)
    caida_mult = np.array([ESCENARIOS[k].get("caida", 1.0) for k in esc])
    score = (edad / 30) * caida_mult + rng.normal(0, .7, n)
    riesgo = np.digitize(score, [1.2, 2.2, 3.0]) + 1
    riesgo = pd.array(riesgo, dtype="Int64")      # entero con vacíos (se escribe 1, 2, 3, 4 y no 1.0)
    riesgo[rng.random(n) < .08] = pd.NA

    # ---------------- motivo, medio de llegada, tipo de atención
    mot = np.empty(n, dtype=object)
    txt = np.empty(n, dtype=object)
    for k, e in ESCENARIOS.items():
        m = np.where(esc == k)[0]
        mot[m] = rng.choice(e["mot"], len(m))
        plant = rng.choice(e["txt"], len(m))
        txt[m] = [p.format(h=rng.integers(1, 12), d=rng.integers(1, 7), n=rng.integers(2, 8), s=rng.integers(12, 40))
                  for p in plant]
    mot[rng.random(n) < .30] = "OTROS, ESPECIFICAR"   # en la base real se usa mucho "OTROS"
    txt[rng.random(n) < .01] = None

    p_samu = np.array([ESCENARIOS[k].get("samu", 0) for k in esc]) + np.select([cat == 1, cat == 2], [.5, .15], 0)
    p_carab = np.array([ESCENARIOS[k].get("carabineros", 0) for k in esc])
    r = rng.random(n)
    medio = np.where(r < np.clip(p_samu, 0, .95),
                     rng.choice(["SAMU MOVIL AVANZADO", "SAMU MOVIL BASICO"], n),
                     np.where(r < p_samu + p_carab, "CARABINEROS",
                              rng.choice(["POR SUS PROPIOS MEDIOS", "AMBULANCIA APS", "SAMU MOVIL BASICO"], n,
                                         p=[.88, .08, .04])))
    tipo_at = np.where(esc == "dental", "ADONTOLOGICO", "ADULTO")

    # ---------------- edad y unidad (A/M/D)
    edad_tipo = np.full(n, "A", dtype=object)
    edad_val = np.floor(edad).astype(int)
    meses = edad < 1
    edad_tipo[meses] = "M"
    edad_val[meses] = np.floor(edad[meses] * 12).astype(int)
    dias_m = meses & (edad_val == 0)
    edad_tipo[dias_m] = "D"
    edad_val[dias_m] = rng.integers(1, 30, dias_m.sum())

    # ---------------- desenlace (posterior al triage)
    def destino(c, k, e):
        if k == "agresion" and rng.random() < .3:
            return "SERVICIO MEDICO LEGAL"
        if c == 1:
            return rng.choice(["UCI ADULTO" if e >= 15 else "UCI PEDIATRICA", "HOSPITALIZACION UTI",
                               "HOSPITALIZACION MEDICINA", "TRASLADO RED PUBLICA", "ANATOMIA PATOLOGICA"],
                              p=[.4, .2, .25, .1, .05])
        if c == 2:
            return rng.choice(["HOSPITALIZACION MEDICINA", "HOSPITALIZACION CIRUGIA", "URGENCIA OBSERVACION",
                               "DOMICILIO", "HOSPITALIZAR", "UCI ADULTO"], p=[.25, .12, .15, .35, .08, .05])
        if c == 3:
            return rng.choice(["DOMICILIO", "HOSPITALIZACION MEDICINA", "HOSPITALIZACION CIRUGIA",
                               "URGENCIA OBSERVACION", "CONSULTORIO", "POLI TRAUMATOLOGIA"],
                              p=[.70, .08, .05, .07, .06, .04])
        return rng.choice(["DOMICILIO", "CONSULTORIO", "A.P.S.", "POLI TRAUMATOLOGIA", "HOSPITALIZACION MEDICINA",
                           "SIN CONSIGNAR"], p=[.80, .07, .05, .04, .02, .02])

    dest = np.array([destino(c, k, e) for c, k, e in zip(cat, esc, edad)], dtype=object)
    dest[(esc == "gineco") & (cat <= 3) & (rng.random(n) < .3)] = "HOSPITALIZA OBSTETRICIA"
    dest[(edad < 15) & np.char.startswith(dest.astype(str), "HOSPITALIZACION") & (rng.random(n) < .8)] = \
        "HOSPITALIZACION PEDIATRIA"

    p_grave = np.array([0, .62, .28, .06, .015, .005])[cat]
    p_med = np.array([0, .30, .45, .32, .15, .05])[cat]
    u = rng.random(n)
    pron = np.where(u < p_grave, "GRAVE", np.where(u < p_grave + p_med, "MEDIANA GRAVEDAD", "LEVE")).astype(object)
    salida = np.where(rng.random(n) < .025 * (cat >= 4), "ABANDONO", "ATENDIDO").astype(object)

    df = pd.DataFrame({
        "urg_folio": np.arange(n) + 2300001,
        "urg_fecha_hora": urg.strftime("%Y-%m-%d %H:%M:%S"),
        "tri_fecha_hora": tri.strftime("%Y-%m-%d %H:%M:%S"),
        "tri_dm": tri_dm, "tri_alergia": tri_alergia, "tri_alergia_detalle": detalle,
        "tri_lcfa": tri_lcfa, "tri_vb": tri_vb, "tri_hta": tri_hta,
        "tri_ggw": vacio(ggw, .03), "tri_eva": vacio(eva, .08),
        "tri_f_respiratoria": vacio(np.round(fr), .04), "tri_pulso": vacio(np.round(pulso), .02),
        "tri_hgt": np.round(hgt), "tri_temperatura": vacio(temp, .03), "riesgo_de_caida": riesgo,
        "tri_peso": vacio(np.round(peso), .25),
        "tri_presion_minima": vacio(np.round(pas), .03),   # SISTÓLICA según el diccionario
        "tri_presion_maxima": vacio(np.round(pad), .03),   # DIASTÓLICA según el diccionario
        "tri_sato2": vacio(np.round(sato2), .03),
        "tri_motivo_cons": txt,
        "niv_cod": np.char.add("C", cat.astype(str)), "sex_cod": sexo,
        "pac_edad": edad_val, "pac_edad_tipo": edad_tipo,
        "glosa_tipo_atencion": tipo_at, "mot_glosa": mot, "med_glosa": medio,
        "dest_glosa": dest, "pron_glosa": pron, "glosa_salida": salida,
    })
    df = df.sort_values("urg_fecha_hora").reset_index(drop=True)
    df["urg_folio"] = np.arange(len(df)) + 2300001

    # ---------------- errores de registro típicos (para probar la limpieza)
    idx = lambda k: rng.choice(len(df), k, replace=False)
    df.loc[idx(150), "tri_pulso"] = 0                                    # 0 = no medido
    i = idx(80)                                                          # presiones invertidas
    df.loc[i, ["tri_presion_minima", "tri_presion_maxima"]] = df.loc[i, ["tri_presion_maxima", "tri_presion_minima"]].values
    i = idx(40)                                                          # triage antes de la admisión
    df.loc[i, "tri_fecha_hora"] = (pd.to_datetime(df.loc[i, "urg_fecha_hora"]) - pd.Timedelta("5min")).dt.strftime(
        "%Y-%m-%d %H:%M:%S")
    df.loc[idx(30), "tri_temperatura"] = 3.6                             # coma decimal mal digitada (36 -> 3.6)
    df = pd.concat([df, df.sample(25, random_state=semilla)]).sort_values("urg_fecha_hora")   # duplicados
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=50000, help="número de atenciones")
    ap.add_argument("--inicio", default="2023-01-01")
    ap.add_argument("--fin", default="2024-12-31")
    ap.add_argument("--semilla", type=int, default=2024)
    ap.add_argument("--salida", default="datos_urgencia_sinteticos.txt")
    a = ap.parse_args()
    df = generar(a.n, a.inicio, a.fin, a.semilla)
    df.to_csv(a.salida, sep="\t", index=False, encoding="utf-8")
    print(f"{len(df):,} filas -> {a.salida}")


if __name__ == "__main__":
    main()
