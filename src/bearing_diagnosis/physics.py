import numpy as np

def bearing_frequencies(rpm, rolling_elements=9, ball_diameter=.3126, pitch_diameter=1.537, contact_angle_deg=0):
    """Return ideal kinematic frequencies in Hz; diameters share one unit."""
    values=[rpm,rolling_elements,ball_diameter,pitch_diameter,contact_angle_deg]
    if not np.isfinite(values).all() or rpm<=0 or rolling_elements<2 or int(rolling_elements)!=rolling_elements:
        raise ValueError('Invalid rotation speed or rolling element count.')
    if not 0<ball_diameter<pitch_diameter or not 0<=contact_angle_deg<90:
        raise ValueError('Invalid bearing geometry.')
    fr=rpm/60; ratio=ball_diameter/pitch_diameter; c=ratio*np.cos(np.deg2rad(contact_angle_deg))
    return {'FTF':fr/2*(1-c),'BPFO':rolling_elements*fr/2*(1-c),
            'BPFI':rolling_elements*fr/2*(1+c),'BSF':fr/(2*ratio)*(1-c*c)}
