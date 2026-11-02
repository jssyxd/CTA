# Audited Tree and Anchor Inference Engine
import numpy as np

class AuditedCtaPredictor:
    """
    Exact recreation of the audited 4-tree GBDT + Linear Anchor additive model.
    """
    @staticmethod
    def predict_one(f):
        F00 = f.get('F00')
        F01 = f.get('F01')
        F02 = f.get('F02')
        F03 = f.get('F03')
        F04 = f.get('F04')
        F05 = f.get('F05')
        F06 = f.get('F06')
        F07 = f.get('F07')
        F08 = f.get('F08')
        F09 = f.get('F09')
        F10 = f.get('F10')
        F11 = f.get('F11')
        F12 = f.get('F12')

        # Tree 0
        t0n6 = 0.006682644545670509 if (F05 is None or np.isnan(F05) or F05 <= 0.12554645109009693) else 0.020468408603572092
        t0n5 = 0.010571839435716894 if (F01 is None or np.isnan(F01) or F01 <= 0.08226351603910521) else -0.0047029605558514655
        t0n4 = t0n6 if (F07 is not None and F07 in [0, 2, 8, 14, 18, 20, 22]) else t0n5
        t0n2 = 0.0039473075719087085 if (F03 is not None and not np.isnan(F03) and F03 <= -0.3485282836817051) else -0.0007024039490765196
        t0n3 = 0.0031816841149915104 if (F12 is not None and not np.isnan(F12) and F12 <= -0.9330127018922192) else -0.005769471342494961
        t0n1 = t0n2 if (F11 is None or np.isnan(F11) or F11 <= 44.5) else t0n3
        t0n0 = t0n4 if (F04 is not None and not np.isnan(F04) and F04 <= -1.1733550604278395) else t0n1

        # Tree 1
        t1n5 = -0.0017688611501784246 if (F05 is None or np.isnan(F05) or F05 <= 0.28387456975021563) else 0.016050493859195197
        t1n6 = 0.018208961684020177 if (F09 is None or np.isnan(F09) or F09 <= 0.002273686965388912) else 0.006445394560088649
        t1n4 = t1n5 if (F06 is not None and not np.isnan(F06) and F06 <= -0.006827964183697319) else t1n6
        t1n2 = -0.0010027191191736172 if (F07 is not None and F07 in [0, 2, 5, 9, 10, 12, 13, 18, 20, 21, 22, 24]) else 0.005768726379502316
        t1n3 = 0.0010793481027867107 if (F07 is not None and F07 in [0, 9, 10, 12, 14, 21, 24, 25]) else -0.0024302368313728917
        t1n1 = t1n2 if (F00 is not None and not np.isnan(F00) and F00 <= -4810.5) else t1n3
        t1n0 = t1n4 if (F04 is not None and not np.isnan(F04) and F04 <= -1.1733550604278395) else t1n1

        # Tree 2
        t2n5 = -0.001729492701248419 if (F05 is None or np.isnan(F05) or F05 <= 0.28387456975021563) else 0.01589075182422991
        t2n6 = 0.01794331059282608 if (F09 is None or np.isnan(F09) or F09 <= 0.002273686965388912) else 0.00630175968678377
        t2n4 = t2n5 if (F06 is not None and not np.isnan(F06) and F06 <= -0.006827964183697319) else t2n6
        t2n2 = 0.005840272185940179 if (F07 is not None and F07 in [1, 3, 4, 6, 7, 8, 11, 15, 16, 17, 19, 23, 25]) else -0.0007415103452858813
        t2n3 = 0.001054014249988056 if (F07 is not None and F07 in [0, 9, 10, 12, 14, 21, 24, 25]) else -0.0023711228657924597
        t2n1 = t2n2 if (F00 is not None and not np.isnan(F00) and F00 <= -4810.5) else t2n3
        t2n0 = t2n4 if (F04 is not None and not np.isnan(F04) and F04 <= -1.1733550604278395) else t2n1

        # Tree 3
        t3n4 = -0.0016910004860372636 if (F05 is None or np.isnan(F05) or F05 <= 0.28387456975021563) else 0.015730162570219433
        t3n6 = 0.004806843132844332 if (F12 is None or np.isnan(F12) or F12 <= 1e-35) else 0.01686998387808037
        t3n3 = t3n4 if (F06 is not None and not np.isnan(F06) and F06 <= -0.006827964183697319) else t3n6
        t3n2 = 0.00792234074123823 if (F03 is None or np.isnan(F03) or F03 <= -0.3612985813469756) else 0.0007696984220099653
        t3n5 = 0.0017208096157623257 if (F10 is None or np.isnan(F10) or F10 <= 0.7491414048201174) else -0.001841112021263396
        t3n1 = t3n2 if (F00 is not None and not np.isnan(F00) and F00 <= -4810.5) else t3n5
        t3n0 = t3n3 if (F04 is not None and not np.isnan(F04) and F04 <= -1.1733550604278395) else t3n1

        # Anchor
        a0 = 0.1934327191040415 if (F02 is None or np.isnan(F02)) else F02
        a1 = 0.260373401455944 if (F03 is None or np.isnan(F03)) else F03
        a2 = 0.04037692015393009 if (F08 is None or np.isnan(F08)) else F08
        a3 = 0.053886504530281355 if (F05 is None or np.isnan(F05)) else F05

        m3 = 1.0 if (F03 is None or np.isnan(F03)) else 0.0
        m8 = 1.0 if (F08 is None or np.isnan(F08)) else 0.0

        anchor = 0.003370711100079098 + 0.22085202491770237 * a0 - 0.025440510686271876 * a1 + 0.011081234307655832 * a2 - 0.8794849773222955 * a3 - 1.5484110831754383 * m3 - 0.12714695841134904 * m8

        return t0n0 + t1n0 + t2n0 + t3n0 + anchor
