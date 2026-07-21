from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import queue
import time
import os
import logging

app = Flask(__name__)
CORS(app)  # Allow web page to communicate with this server

# Disable Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app.logger.setLevel(logging.ERROR)

# Global order queue (shared between web and drone)
order_queue = queue.Queue()
active_orders = []
current_order = None

# Store building mapping
STORE_MAP = {
    'B': 'Food Store',
    'C': 'Pharmacy',
    'D': 'Grocery Store'
}

@app.route('/')
def home():
    # Serve the main HTML interface
    try:
        html_path = os.path.join(os.path.dirname(__file__), 'delivery_app_interface.html')
        if os.path.exists(html_path):
            return send_file(html_path)
        else:
            return 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/delivery_app_interface.html')
def serve_interface():
    # Alternative route for the HTML interface
    return home()

@app.route('/place_order', methods=['POST'])
def place_order():
    # Receive order from web interface
    try:
        order_data = request.json
        
        # Add order to queue
        order = {
            'product': order_data['product'],
            'location': order_data['location'],
            'timestamp': order_data['timestamp'],
            'status': 'pending',
            'state': 'Waiting in queue'
        }
        
        order_queue.put(order)
        active_orders.append(order)
        
        print(f"📦 New order received: {order['product']['name']} → Building {order['location']}")
        
        return jsonify({
            'success': True,
            'message': 'Order placed successfully',
            'order': order
        })
    
    except Exception as e:
        print(f"Error placing order: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/order_status', methods=['GET'])
def order_status():
    # Send current order status to web interface
    return jsonify({
        'orders': active_orders,
        'current': current_order
    })


@app.route('/update_order_status', methods=['POST'])
def update_order_status():
    # Update order status from drone simulation
    try:
        data = request.json
        order_id = data.get('order_id', 0)
        new_status = data.get('status')
        new_state = data.get('state')
        
        if order_id < len(active_orders):
            active_orders[order_id]['status'] = new_status
            active_orders[order_id]['state'] = new_state
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def get_next_order():
    # Get next order from queue (called by drone simulation)
    global current_order
    try:
        if not order_queue.empty():
            current_order = order_queue.get_nowait()
            return current_order
        return None
    except:
        return None


def update_current_order_status(status, state):
    # Update status of currently processing order
    if current_order and current_order in active_orders:
        idx = active_orders.index(current_order)
        active_orders[idx]['status'] = status
        active_orders[idx]['state'] = state


def run_flask():
    # Run Flask server in a separate thread
    print("🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    run_flask()