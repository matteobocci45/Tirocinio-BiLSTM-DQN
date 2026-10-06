from sklearn.preprocessing import LabelEncoder
import numpy as np



# ============================================================
# SAFE LABEL ENCODER
# ============================================================

class SafeLabelEncoder:


    def __init__(self):

        self.encoder = LabelEncoder()

        self.fitted = False



    def fit(self, values):


        values = [

            str(v)
            if v is not None
            else "UNKNOWN"

            for v in values

        ]


        # aggiungiamo sempre UNKNOWN

        values.append(
            "UNKNOWN"
        )


        self.encoder.fit(
            values
        )


        self.fitted=True



    def transform(self, values):


        if not self.fitted:

            raise RuntimeError(
                "Encoder non inizializzato"
            )



        result=[]


        known = set(
            self.encoder.classes_
        )



        for value in values:


            if value is None:

                value="UNKNOWN"


            value=str(
                value
            )



            if value not in known:

                value="UNKNOWN"



            result.append(

                self.encoder.transform(
                    [value]
                )[0]

            )


        return np.array(
            result
        )



    def transform_one(self,value):

        return self.transform(
            [value]
        )[0]



    def inverse_transform(self,value):

        return self.encoder.inverse_transform(
            [value]
        )[0]



# ============================================================
# BUILD ALL ENCODERS
# ============================================================

def fit_encoders(logs):


    encoders={}



    fields={


        "event":
            "event_type",



        "source":
            "source",



        "role":
            "role",



        "resource":
            "resource"

    }



    for name, field in fields.items():


        encoder = SafeLabelEncoder()



        encoder.fit(

            [

                log.get(field)

                for log in logs

            ]

        )


        encoders[name]=encoder



    return encoders



# ============================================================
# SAVE / LOAD
# ============================================================

import pickle



def save_encoders(
        encoders,
        path="encoders.pkl"
):


    with open(
        path,
        "wb"
    ) as f:


        pickle.dump(
            encoders,
            f
        )



def load_encoders(
        path="encoders.pkl"
):


    with open(
        path,
        "rb"
    ) as f:


        return pickle.load(
            f
        )