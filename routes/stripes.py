from flask import Blueprint, render_template, url_for, request, abort
import stripe 
from ..app import app


appStripe = Blueprint('routes-stripe', __name__)


app.config['STRIPE_PUBLIC_KEY'] = 'pk_test_51LlfNGGuGVj1FqA7CozX49ZPxH2D0wtcqot4eKakjXMfSeUjHfkCbcy46RS3lz3KGEHu6yCZw6GUuwBWFD7h5yhm00MxmoX9ht'
app.config['STRIPE_SECRET_KEY'] = 'sk_test_51LlfNGGuGVj1FqA7hahRvMxeD4tdvTY5XmodFXqKJx48gotl0petE9pd3AIR4BbCAG0ExlsKN8yQXe8UxLHbcWXT000WJnDsB6'

stripe.api_key = app.config['STRIPE_SECRET_KEY']

@appStripe.route('/stripe')
def stripe():
    '''
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price': 'price_1GtKWtIdX0gthvYPm4fJgrOr',
            'quantity': 1,
        }],
        mode='payment',
        success_url=url_for('thanks', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
        cancel_url=url_for('index', _external=True),
    )
    '''
    return render_template(
        'stripe.html', 
        #checkout_session_id=session['id'], 
        #checkout_public_key=app.config['STRIPE_PUBLIC_KEY']
    )

@appStripe.route('/stripe_pay')
def stripe_pay():
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price': 'price_1LliYVGuGVj1FqA7XqmrYehi',
            'quantity': 1,
        }],
        mode='payment',
        success_url=url_for('thanks', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
        cancel_url=url_for('stripe', _external=True),
    )
    return {
        'checkout_session_id': session['id'], 
        'checkout_public_key': app.config['STRIPE_PUBLIC_KEY']
    }

@appStripe.route('/thanks')
def thanks():
    return render_template('thanks.html')

@appStripe.route('/stripe_webhook', methods=['POST'])
def stripe_webhook():
    print('WEBHOOK CALLED')

    if request.content_length > 1024 * 1024:
        print('REQUEST TOO BIG')
        abort(400)
    payload = request.get_data()
    sig_header = request.environ.get('HTTP_STRIPE_SIGNATURE')
    endpoint_secret = 'YOUR_ENDPOINT_SECRET'
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        print('INVALID PAYLOAD')
        return {}, 400
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        print('INVALID SIGNATURE')
        return {}, 400

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        print(session)
        line_items = stripe.checkout.Session.list_line_items(session['id'], limit=1)
        print(line_items['data'][0]['description'])
    return {}