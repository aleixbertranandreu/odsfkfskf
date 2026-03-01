import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useCart } from '../context/CartContext';
import './Checkout.css';

export default function Checkout() {
    const { items, totalPrice, clearCart, totalItems } = useCart();
    const [submitted, setSubmitted] = useState(false);

    const shipping = totalPrice >= 50 ? 0 : 4.95;
    const total = totalPrice + shipping;

    const handleSubmit = (e) => {
        e.preventDefault();
        setSubmitted(true);
        clearCart();
    };

    if (submitted) {
        return (
            <main className="checkout-page">
                <div className="checkout-success fade-in">
                    <div className="checkout-success-icon">✓</div>
                    <h2>¡Pedido confirmado!</h2>
                    <p>Gracias por tu compra. Recibirás un email de confirmación en breve.</p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
                        Nº de pedido: MOD-{Date.now().toString(36).toUpperCase()}
                    </p>
                    <Link to="/" className="btn btn-primary">
                        Volver al inicio
                    </Link>
                </div>
            </main>
        );
    }

    if (totalItems === 0) {
        return (
            <main className="checkout-page">
                <div className="checkout-success">
                    <h2>No hay artículos</h2>
                    <p>Añade productos al carrito antes de proceder al pago.</p>
                    <Link to="/catalogo" className="btn btn-primary" style={{ marginTop: '1.5rem' }}>
                        Ver catálogo
                    </Link>
                </div>
            </main>
        );
    }

    return (
        <main className="checkout-page">
            <h1>Checkout</h1>

            <div className="checkout-layout fade-in">
                {/* Form */}
                <form className="checkout-form" onSubmit={handleSubmit}>
                    <div className="checkout-section">
                        <h3>Datos de envío</h3>
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="firstName">Nombre</label>
                                <input id="firstName" type="text" required placeholder="María" />
                            </div>
                            <div className="form-group">
                                <label htmlFor="lastName">Apellidos</label>
                                <input id="lastName" type="text" required placeholder="García López" />
                            </div>
                        </div>
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="email">Email</label>
                                <input id="email" type="email" required placeholder="maria@ejemplo.com" />
                            </div>
                        </div>
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="address">Dirección</label>
                                <input id="address" type="text" required placeholder="Calle Mayor 12, 3ºA" />
                            </div>
                        </div>
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="city">Ciudad</label>
                                <input id="city" type="text" required placeholder="Madrid" />
                            </div>
                            <div className="form-group">
                                <label htmlFor="zip">Código postal</label>
                                <input id="zip" type="text" required placeholder="28001" />
                            </div>
                        </div>
                    </div>

                    <div className="checkout-section">
                        <h3>Método de pago</h3>
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="card">Número de tarjeta</label>
                                <input id="card" type="text" required placeholder="1234 5678 9012 3456" />
                            </div>
                        </div>
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="expiry">Caducidad</label>
                                <input id="expiry" type="text" required placeholder="MM/AA" />
                            </div>
                            <div className="form-group">
                                <label htmlFor="cvv">CVV</label>
                                <input id="cvv" type="text" required placeholder="123" />
                            </div>
                        </div>
                    </div>

                    <button className="btn btn-primary checkout-submit" type="submit">
                        Confirmar pedido — {total.toFixed(2)} EUR
                    </button>
                </form>

                {/* Summary */}
                <div className="checkout-summary">
                    <h3>Resumen del pedido</h3>
                    {items.map((item) => (
                        <div key={`${item.id}-${item.size}`} className="checkout-summary-item">
                            <img src={item.image} alt={item.name} />
                            <div className="checkout-summary-item-info">
                                <p>{item.name}</p>
                                <p className="meta">Talla: {item.size} · Cant: {item.quantity}</p>
                                <p className="item-price">{(item.price * item.quantity).toFixed(2)} EUR</p>
                            </div>
                        </div>
                    ))}

                    <div className="checkout-totals">
                        <div className="cart-summary-row">
                            <span>Subtotal</span>
                            <span>{totalPrice.toFixed(2)} EUR</span>
                        </div>
                        <div className="cart-summary-row">
                            <span>Envío</span>
                            <span>{shipping === 0 ? 'Gratis' : `${shipping.toFixed(2)} EUR`}</span>
                        </div>
                        <div className="cart-summary-row total">
                            <span>Total</span>
                            <span>{total.toFixed(2)} EUR</span>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
